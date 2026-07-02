#!/usr/bin/env python3
"""Humanmade — a human simulator with a local AI brain.

Run:  python main.py
The human lives on its own clock (default: 1 real second = 1 sim minute).
It thinks, survives, and will talk to you first. Type to talk back.

Commands:
  /status        show vitals            /memories [n]  show recent memories
  /speed <n>     sim minutes per real second (default 1, max 60)
  /restock       buy groceries          /thoughts      toggle inner monologue
  /away          tell it you're leaving (it waits; type anything to return)
  /plan          see today's plan        /dream        recall last night's dream
  /newlife       start a new person     /quit          save and exit
"""

import json
import os
import sys
import threading
import time
import traceback

from humanmade.agent import Human

STATE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "state")
CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")

C_SPEAK = "\033[96m"   # cyan   — spoken words
C_EVENT = "\033[90m"   # grey   — narration
C_THOUGHT = "\033[35m" # purple — inner monologue
C_YOU = "\033[93m"
C_RESET = "\033[0m"

print_lock = threading.Lock()
show_thoughts = True


def load_config() -> dict:
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH) as f:
            return json.load(f)
    return {}


def out(text: str) -> None:
    with print_lock:
        sys.stdout.write("\r\033[K" + text + "\nyou> ")
        sys.stdout.flush()


def main() -> None:
    config = load_config()
    speed = float(config.get("speed", 1.0))  # sim minutes per real second

    human = Human(
        STATE_DIR, config,
        on_speak=lambda t: out(f"{C_SPEAK}{human.persona['name']}: {t}{C_RESET}"),
        on_event=lambda t: out(f"{C_EVENT}· {t}{C_RESET}"),
        on_thought=lambda t: show_thoughts and out(f"{C_THOUGHT}({t}){C_RESET}"),
    )

    name = human.persona["name"]
    print(__doc__)
    print(f"{C_EVENT}LLM brain: "
          + (f"online ({human.llm.provider}/{human.llm.model})" if human.llm_online
             else f"OFFLINE — reflex survival mode. Start your local model "
                  f"(e.g. `ollama run {human.llm.model}`) and it will reconnect.")
          + C_RESET)
    print(f"{C_EVENT}{name}, {human.persona['age']} — {human.persona['personality']} "
          f"({human.persona.get('chronotype', 'intermediate')}). "
          f"{human.memory.count()} memories on record.{C_RESET}\n")

    stop = threading.Event()

    def life_loop() -> None:
        last = time.monotonic()
        last_save = last
        while not stop.is_set():
            time.sleep(1.0)
            now = time.monotonic()
            try:
                if human.body.alive:
                    human.tick((now - last) * speed)
                if now - last_save > 60:
                    human.save()
                    last_save = now
            except Exception:
                with print_lock:
                    traceback.print_exc()
                out(f"{C_EVENT}!! simulation hiccup (see traceback above) — "
                    f"life continues{C_RESET}")
            last = now

    sim = threading.Thread(target=life_loop, daemon=True)
    sim.start()

    global show_thoughts
    try:
        while True:
            try:
                line = input("you> ").strip()
            except EOFError:
                break
            if not line:
                continue
            if line == "/quit":
                break
            elif line == "/status":
                with print_lock:
                    print("\n".join(human.status_report()))
            elif line.startswith("/memories"):
                parts = line.split()
                n = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 15
                memories = human.recent_memories(n)
                with print_lock:
                    for m in memories:
                        d, mm = divmod(int(m.sim_minutes), 1440)
                        print(f"  [d{d} {mm//60:02d}:{mm%60:02d}] "
                              f"({m.kind}, imp {m.importance:.0f}) {m.text}")
            elif line.startswith("/speed"):
                parts = line.split()
                if len(parts) > 1:
                    speed = max(0.1, min(60.0, float(parts[1])))
                print(f"speed: {speed} sim-min per real second")
            elif line == "/restock":
                total = human.restock()
                out(f"{C_EVENT}· groceries delivered — fridge now holds {total} portions{C_RESET}")
            elif line == "/thoughts":
                show_thoughts = not show_thoughts
                print(f"inner monologue: {'on' if show_thoughts else 'off'}")
            elif line == "/away":
                human.set_away()
                print(f"{C_EVENT}You step out. {name} is now on their own — "
                      f"type anything when you're back.{C_RESET}")
            elif line == "/plan":
                with print_lock:
                    if human.today_plan:
                        print(f"{name}'s plan for today:")
                        for item in human.today_plan:
                            print(f"  • {item}")
                    else:
                        print(f"{name} hasn't made a plan yet "
                              "(plans form on waking, with the LLM brain online).")
            elif line == "/dream":
                with print_lock:
                    if human.last_dream:
                        print(f"{name}'s last dream: {human.last_dream}")
                    else:
                        print(f"{name} doesn't remember dreaming yet.")
            elif line == "/newlife":
                if human.body.alive:
                    print(f"{name} is still alive. This only works after death.")
                else:
                    human.new_life()
                    name = human.persona["name"]
                    print(f"A new person opens their eyes: {name}, "
                          f"{human.persona['age']} — {human.persona['personality']}.")
            elif line.startswith("/"):
                print(__doc__)
            else:
                with print_lock:
                    sys.stdout.write(f"\033[F\033[K{C_YOU}you: {line}{C_RESET}\n")
                human.hear(line)
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        human.save()
        print(f"\nsaved. {name} keeps existing between runs — memories and body "
              f"state persist in {STATE_DIR}/")


if __name__ == "__main__":
    main()
