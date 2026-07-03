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
  /journal [n]   read its diary          /medicine     order medicine when sick
  /bond          how it feels about you  /habits       its learned routine
  /web [n]       what it's read online   /read         its work in progress
  /works         its finished creations  /biography    its whole life story
  /newlife       start a new person     /quit          save and exit
"""

import json
import os
import sys
import threading
import time
import traceback
import urllib.parse
import urllib.request

from humanmade import timeflow
from humanmade.agent import Human

WEATHER_POLL_SECONDS = 1800.0   # 30 real minutes

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


def fetch_real_weather(latitude: float, longitude: float,
                       timeout: float = 5.0) -> str | None:
    """One reading from Open-Meteo (no API key needed), mapped onto this
    sim's weathers. Any failure — offline, timeout, bad response — falls
    back silently to None so the simulated weather just keeps cycling."""
    try:
        url = ("https://api.open-meteo.com/v1/forecast?" +
               urllib.parse.urlencode({"latitude": latitude,
                                       "longitude": longitude,
                                       "current_weather": "true"}))
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            data = json.load(resp)
        code = int(data["current_weather"]["weathercode"])
        return timeflow.wmo_to_weather(code)
    except Exception:
        return None


def _format_span(sim_minutes: float) -> str:
    hours = sim_minutes / 60.0
    if hours < 1:
        return f"{sim_minutes:.0f} minutes"
    if hours < 48:
        return f"{hours:.1f} hours"
    return f"{hours / 24:.1f} days"


def main() -> None:
    config = load_config()
    speed = float(config.get("speed", 1.0))  # sim minutes per real second
    time_config = config.get("time", {})
    real_time_mode = str(time_config.get("mode", "sim")).lower() == "real"
    weather_config = time_config.get("weather", {})
    real_weather_enabled = bool(weather_config.get("real", False))
    weather_lat = weather_config.get("latitude")
    weather_lon = weather_config.get("longitude")
    if real_weather_enabled and (weather_lat is None or weather_lon is None):
        print(f"{C_EVENT}Real weather is enabled but latitude/longitude "
              f'are missing from config.json ("time"."weather") — '
              f"falling back to simulated weather.{C_RESET}")
        real_weather_enabled = False

    # first run: let the user name their human (blank = a random person)
    chosen_name = None
    first_run = not os.path.exists(os.path.join(STATE_DIR, "memory.sqlite3"))
    if first_run and sys.stdin.isatty():
        chosen_name = input("A new person is about to exist. "
                            "Name them (or press Enter for a stranger): ").strip() or None

    human = Human(
        STATE_DIR, config,
        on_speak=lambda t: out(f"{C_SPEAK}{human.persona['name']}: {t}{C_RESET}"),
        on_event=lambda t: out(f"{C_EVENT}· {t}{C_RESET}"),
        on_thought=lambda t: show_thoughts and out(f"{C_THOUGHT}({t}){C_RESET}"),
        name=chosen_name,
    )
    human.world.weather_locked = real_weather_enabled  # config is the source of truth each run

    name = human.persona["name"]
    print(__doc__)
    print(f"{C_EVENT}LLM brain: "
          + (f"online ({human.llm.provider}/{human.llm.model})" if human.llm_online
             else f"OFFLINE — reflex survival mode. Start your local model "
                  f"(e.g. `ollama run {human.llm.model}`) and it will reconnect.")
          + C_RESET)
    if human.internet_enabled:
        print(f"{C_EVENT}Internet: on — {human.persona['name']} can browse "
              f"(allowlist: {', '.join(human._internet_config.get('allowlist') or ['en.wikipedia.org', '*.wikipedia.org'])}){C_RESET}")
    if real_weather_enabled:
        print(f"{C_EVENT}Real weather: on — {name}'s sky mirrors "
              f"{weather_lat},{weather_lon}, checked every 30 minutes.{C_RESET}")
    print(f"{C_EVENT}{name}, {human.persona['age']} — {human.persona['personality']} "
          f"({human.persona.get('chronotype', 'intermediate')}). "
          f"{human.memory.count()} memories on record.{C_RESET}\n")

    anchor = None
    if real_time_mode:
        anchor_json = human.memory.get_meta("time_anchor")
        if anchor_json:
            anchor = timeflow.Anchor.from_dict(json.loads(anchor_json))
            gap = timeflow.target_sim_minutes(anchor) - human.body.sim_minutes
            if gap > 1.0 and human.body.alive:
                print(f"{C_EVENT}Catching up on {_format_span(gap)} that passed "
                      f"while this was closed...{C_RESET}")
                human.catch_up(gap)
                if not human.body.alive:
                    print(f"{C_EVENT}{name} did not make it — died of "
                          f"{human.body.cause_of_death} while you were away. "
                          f"Use /newlife to begin again.{C_RESET}")
                else:
                    print(f"{C_EVENT}{name} lived through it. Say something to "
                          f"hear how it went.{C_RESET}")
        else:
            # first time real mode is on: align to the real hour-of-day once,
            # so the human isn't stuck at a jarring hour relative to your clock
            aligned = timeflow.align_hour_of_day(human.body.sim_minutes,
                                                 timeflow.real_hour_of_day())
            if abs(aligned - human.body.sim_minutes) > 1.0:
                human.body.sim_minutes = aligned
            print(f"{C_EVENT}Time-true mode: {name}'s clock is now anchored to "
                  f"yours — /speed doesn't apply.{C_RESET}")
        anchor = timeflow.rebase(human.body.sim_minutes)
        human.memory.set_meta("time_anchor", json.dumps(anchor.to_dict()))
        human.save()

    stop = threading.Event()

    def life_loop() -> None:
        nonlocal anchor
        last = time.monotonic()
        last_save = last
        while not stop.is_set():
            time.sleep(1.0)
            now = time.monotonic()
            try:
                if human.body.alive:
                    if real_time_mode:
                        delta = timeflow.target_sim_minutes(anchor) - human.body.sim_minutes
                        if delta > 0:
                            human.tick(delta)
                    else:
                        human.tick((now - last) * speed)
                if now - last_save > 60:
                    human.save()
                    if real_time_mode:
                        anchor = timeflow.rebase(human.body.sim_minutes)
                        human.memory.set_meta("time_anchor",
                                              json.dumps(anchor.to_dict()))
                    last_save = now
            except Exception:
                with print_lock:
                    traceback.print_exc()
                out(f"{C_EVENT}!! simulation hiccup (see traceback above) — "
                    f"life continues{C_RESET}")
            last = now

    sim = threading.Thread(target=life_loop, daemon=True)
    sim.start()

    if real_weather_enabled:
        def weather_loop() -> None:
            while not stop.is_set():
                weather = fetch_real_weather(weather_lat, weather_lon)
                if weather is not None and human.body.alive:
                    human.set_real_weather(weather)
                if stop.wait(WEATHER_POLL_SECONDS):
                    break
        threading.Thread(target=weather_loop, daemon=True).start()

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
                if real_time_mode:
                    print(f"{name} lives in real time — /speed doesn't apply "
                          '(set "time": {"mode": "sim"} in config.json to change this).')
                else:
                    parts = line.split()
                    if len(parts) > 1:
                        speed = max(0.1, min(60.0, float(parts[1])))
                    print(f"speed: {speed} sim-min per real second")
            elif line == "/restock":
                r = human.restock()
                if r["bought"] > 0:
                    out(f"{C_EVENT}· groceries delivered — {r['bought']} portions "
                        f"bought, fridge holds {r['total']}, {name} has "
                        f"{r['money']:.0f} credits left{C_RESET}")
                else:
                    out(f"{C_EVENT}· the order was declined — {name} can't afford "
                        f"any food ({r['money']:.0f} credits). They need to work.{C_RESET}")
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
            elif line.startswith("/journal"):
                parts = line.split()
                n = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 5
                entries = human.journal_entries(n)
                with print_lock:
                    if entries:
                        for m in entries:
                            d = int(m.sim_minutes // 1440)
                            print(f"  [day {d}] {m.text}")
                    else:
                        print(f"{name}'s diary is still blank "
                              "(entries are written at bedtime, LLM brain online).")
            elif line == "/read":
                work = human.current_work()
                with print_lock:
                    if not work:
                        print(f"{name} hasn't started a project yet.")
                    else:
                        print(f'"{work.display_title}" ({work.kind}) — '
                              f"{len(work.fragments)} piece"
                              f"{'s' if len(work.fragments) != 1 else ''} so far")
                        if work.synopsis:
                            print(f"  so far: {work.synopsis}")
                        if work.fragments:
                            print(f"  most recent piece:\n"
                                  f"  {work.fragments[-1]['text']}")
            elif line == "/works":
                works = human.finished_works()
                with print_lock:
                    if not works:
                        print(f"{name} hasn't finished anything yet.")
                    else:
                        for w in works:
                            d = int((w.finished_sim or 0) // 1440)
                            print(f'  [day {d}] "{w.display_title}" ({w.kind}) — '
                                  f"sold for {w.sale_price:.0f} credits")
            elif line.startswith("/web"):
                parts = line.split()
                n = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 5
                with human.lock:
                    entries = human.memory.recent(n, kinds=("web",))
                with print_lock:
                    if not human.internet_enabled:
                        print(f"{name} doesn't have internet access "
                              '(set "internet": {"enabled": true} in config.json).')
                    elif human._browse_unavailable:
                        print(f"{name} tried to get online, but this computer has "
                              "no working browser (pip install playwright && "
                              "playwright install chromium).")
                    elif entries:
                        for m in entries:
                            d = int(m.sim_minutes // 1440)
                            print(f"  [day {d}] {m.text}")
                    else:
                        print(f"{name} hasn't looked anything up yet.")
            elif line == "/medicine":
                r = human.medicine()
                if r["ok"]:
                    out(f"{C_EVENT}· medicine delivered — fever easing "
                        f"(sickness {r['sickness']:.0f}/100, "
                        f"{r['money']:.0f} credits left){C_RESET}")
                elif r["reason"] == "not sick":
                    print(f"{name} isn't sick right now.")
                else:
                    out(f"{C_EVENT}· the pharmacy declined — {name} can't afford "
                        f"medicine ({r['money']:.0f} credits){C_RESET}")
            elif line == "/bond":
                with print_lock:
                    with human.lock:
                        days = (human.body.sim_minutes
                                - human.bond["first_met_sim"]) / 1440
                        print(f"{name} has known you {days:.1f} days — "
                              f"trust {human.bond['trust']:.0f}/100, closeness "
                              f"{human.bond['closeness']:.0f}/100 "
                              f"({human.bond_level()})")
                        known = human.memory.recent(8, kinds=("companion",))
                        for m in known:
                            print(f"  knows: {m.text}")
            elif line == "/biography":
                status = human.request_biography()
                with print_lock:
                    if status == "offline":
                        print(f"{name}'s mind needs to be online to write this.")
                    elif status == "writing":
                        print(f"{name} is thinking it over — ask again in a moment.")
                    else:
                        print(human.biography_text())
            elif line == "/habits":
                with print_lock:
                    with human.lock:
                        any_habit = False
                        for act in human.HABIT_ACTIONS:
                            hours = human.habitual_hours(act)
                            if hours:
                                any_habit = True
                                times = ", ".join(f"{h:02d}:00-ish" for h in hours)
                                print(f"  usually {act}s around {times}")
                        feels = human._experience_hints()
                        for f in feels:
                            print(f"  {f}")
                        if not any_habit and not feels:
                            print(f"{name} hasn't settled into a routine yet — "
                                  "habits take days of repetition to form.")
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
        if real_time_mode and human.body.alive:
            anchor = timeflow.rebase(human.body.sim_minutes)
            human.memory.set_meta("time_anchor", json.dumps(anchor.to_dict()))
        human.save()
        print(f"\nsaved. {name} keeps existing between runs — memories and body "
              f"state persist in {STATE_DIR}/")


if __name__ == "__main__":
    main()
