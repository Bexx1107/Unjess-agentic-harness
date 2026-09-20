"""Schedule Dashboard Component — interactive UI for managing scheduled subagents and inspecting run logs."""

import time
from typing import Any, Callable, Optional
from nicegui import ui

from unjess.scheduler import Scheduler, ScheduledTask
from unjess.subagents.scheduled_runner import load_task_run_logs


def render_schedule_dashboard(scheduler: Scheduler) -> None:
    """Render the Schedule Dashboard UI component."""
    with ui.card().classes("w-full bg-[#18181c] border border-gray-800 p-4 rounded-xl flex flex-col gap-3 mt-4"):
        # Header
        with ui.row().classes("w-full items-center justify-between border-b border-gray-800 pb-3"):
            with ui.row().classes("items-center gap-2"):
                ui.label("⏱️ SCHEDULED SUBAGENTS & AUTOMATION").classes("text-sm font-bold text-white tracking-wide")
                badge_count = len(scheduler.list_tasks())
                ui.label(f"{badge_count} tasks").classes("text-xs bg-violet-950 text-violet-300 border border-violet-800 px-2 py-0.5 rounded-full")

            with ui.row().classes("gap-2"):
                def _open_create_dialog() -> None:
                    _create_schedule_modal(scheduler, on_saved=render_task_list)

                ui.button("New Schedule", icon="add", on_click=_open_create_dialog).props(
                    "unelevated dense no-caps"
                ).classes("bg-violet-600 hover:bg-violet-500 text-white font-medium px-3 rounded-lg text-xs")

        # Task List Container
        task_container = ui.column().classes("w-full gap-3 mt-1")

        def render_task_list() -> None:
            task_container.clear()
            tasks = scheduler.list_tasks()

            if not tasks:
                with task_container:
                    with ui.column().classes("w-full items-center justify-center p-6 text-center border border-dashed border-gray-800 rounded-lg"):
                        ui.icon("event_note", size="32px").classes("text-gray-600 mb-1")
                        ui.label("No scheduled tasks yet").classes("text-xs font-semibold text-gray-400")
                        ui.label("Schedule headless subagents to run background prompts periodically.").classes("text-[11px] text-gray-500 mt-0.5")
                return

            with task_container:
                for task in tasks:
                    _render_task_card(task, scheduler, on_update=render_task_list)

        render_task_list()


def _render_task_card(task: ScheduledTask, scheduler: Scheduler, on_update: Callable[[], None]) -> None:
    """Render a single scheduled task card."""
    with ui.card().classes("w-full bg-[#222228] border border-gray-800 p-3.5 rounded-lg flex flex-col gap-2"):
        with ui.row().classes("w-full items-center justify-between no-wrap"):
            # Left: Badges & ID
            with ui.row().classes("items-center gap-2 wrap sm:no-wrap"):
                type_color = "bg-amber-950 text-amber-300 border-amber-800" if task.task_type == "cron" else "bg-sky-950 text-sky-300 border-sky-800"
                ui.label(task.task_type.upper()).classes(f"text-[10px] font-bold px-2 py-0.5 rounded border {type_color}")

                ui.label(task.id).classes("text-xs font-mono font-bold text-white")

                # Status Badge
                status_styles = {
                    "running": "bg-blue-950 text-blue-300 border-blue-800",
                    "succeeded": "bg-emerald-950 text-emerald-300 border-emerald-800",
                    "failed": "bg-rose-950 text-rose-300 border-rose-800",
                    "paused": "bg-gray-800 text-gray-400 border-gray-700",
                    "pending": "bg-indigo-950 text-indigo-300 border-indigo-800",
                }
                st_class = status_styles.get(task.last_status, "bg-gray-800 text-gray-400 border-gray-700")
                st_label = task.last_status.upper() if task.enabled else "PAUSED"
                ui.label(st_label).classes(f"text-[10px] font-bold px-2 py-0.5 rounded border {st_class}")

            # Right: Quick Action Buttons
            with ui.row().classes("items-center gap-1.5"):
                def _trigger() -> None:
                    scheduler.trigger_now(task.id)
                    ui.notify(f"⚡ Triggered {task.id} on-demand", type="info", position="top-right")
                    on_update()

                def _toggle_pause() -> None:
                    if task.enabled:
                        scheduler.pause_task(task.id)
                        ui.notify(f"⏸️ Paused {task.id}", type="warning", position="top-right")
                    else:
                        scheduler.resume_task(task.id)
                        ui.notify(f"▶️ Resumed {task.id}", type="positive", position="top-right")
                    on_update()

                def _delete() -> None:
                    scheduler.delete_task(task.id)
                    ui.notify(f"🗑️ Deleted {task.id}", type="negative", position="top-right")
                    on_update()

                def _chat_with_agent() -> None:
                    _open_chat_with_agent_modal(task, scheduler, on_update)

                ui.button(icon="chat", on_click=_chat_with_agent).props("flat dense round").classes("text-sky-400 hover:bg-sky-950").tooltip("Talk to this Scheduled Agent")
                ui.button(icon="play_arrow", on_click=_trigger).props("flat dense round").classes("text-emerald-400 hover:bg-emerald-950").tooltip("Run Now")
                ui.button(icon="pause" if task.enabled else "play_arrow", on_click=_toggle_pause).props("flat dense round").classes("text-amber-400 hover:bg-amber-950").tooltip("Pause / Resume")
                ui.button(icon="history", on_click=_view_logs).props("flat dense round").classes("text-violet-400 hover:bg-violet-950").tooltip("View Run Logs")
                ui.button(icon="delete", on_click=_delete).props("flat dense round").classes("text-rose-400 hover:bg-rose-950").tooltip("Delete Task")

        # Prompt & Timing details
        ui.label(task.prompt).classes("text-xs text-gray-200 bg-[#19191d] p-2 rounded border border-gray-800 break-words font-mono")

        with ui.row().classes("w-full items-center justify-between text-[11px] text-gray-400 mt-0.5"):
            schedule_spec = f"Cron: {task.cron_expression}" if task.task_type == "cron" else f"Timer: {task.duration_seconds:.0f}s"
            last_run_str = time.strftime("%H:%M:%S", time.localtime(task.last_run)) if task.last_run else "Never"
            ui.label(f"⏱️ {schedule_spec}  •  Fired: {task.fire_count}x  •  Last Run: {last_run_str}").classes("text-gray-400 font-mono")


def _open_chat_with_agent_modal(task: ScheduledTask, scheduler: Scheduler, on_update: Callable[[], None]) -> None:
    """Open interactive chat modal to talk directly with a scheduled agent."""
    dialog = ui.dialog()
    with dialog, ui.card().classes("w-[95vw] max-w-[800px] max-h-[90vh] bg-[#18181c] border border-gray-800 p-4 rounded-xl flex flex-col gap-3 text-white"):
        with ui.row().classes("w-full justify-between items-center border-b border-gray-800 pb-2"):
            with ui.row().classes("items-center gap-2"):
                ui.icon("smart_toy", size="20px").classes("text-sky-400")
                ui.label(f"Chat with Agent — {task.id}").classes("text-sm font-bold text-white font-mono")
                type_color = "bg-amber-950 text-amber-300 border-amber-800" if task.task_type == "cron" else "bg-sky-950 text-sky-300 border-sky-800"
                ui.label(task.task_type.upper()).classes(f"text-[10px] font-bold px-2 py-0.5 rounded border {type_color}")
            ui.button(icon="close", on_click=dialog.close).props("flat dense round").classes("text-gray-400")

        # System / Task prompt box
        with ui.card().classes("w-full bg-[#222228] border border-gray-800 p-3 rounded-lg flex flex-col gap-1"):
            ui.label("Current Agent Instructions / Prompt:").classes("text-[11px] font-bold text-sky-400")
            ui.label(task.prompt).classes("text-xs text-gray-300 font-mono break-words")

        # Chat history container
        chat_container = ui.column().classes("w-full max-h-[400px] overflow-y-auto gap-2 p-1 bg-[#141417] rounded-lg border border-gray-800")

        def _render_chat_messages() -> None:
            chat_container.clear()
            runs = load_task_run_logs(task.id)

            if not runs:
                with chat_container:
                    ui.label("No conversation history yet. Send a message below to start chatting!").classes("text-xs text-gray-500 p-4 text-center")
                return

            with chat_container:
                for r in reversed(runs):
                    t_str = time.strftime("%H:%M:%S", time.localtime(r.get("started_at", 0)))
                    # User Prompt bubble
                    with ui.row().classes("w-full justify-end my-1"):
                        with ui.card().classes("max-w-[85%] bg-violet-950/70 border border-violet-800 p-2.5 rounded-xl text-white text-xs"):
                            ui.label("YOU").classes("text-[9px] font-bold text-violet-300 mb-0.5")
                            ui.label(r.get("prompt", "")).classes("font-mono text-gray-200")

                    # Agent Reply bubble
                    with ui.row().classes("w-full justify-start my-1"):
                        with ui.card().classes("max-w-[85%] bg-[#222228] border border-gray-800 p-2.5 rounded-xl text-white text-xs"):
                            status_ico = "✅" if r.get("status") == "succeeded" else "❌"
                            ui.label(f"AGENT {task.id}  •  {t_str} {status_ico}").classes("text-[9px] font-bold text-sky-400 mb-0.5")
                            reply_content = r.get("result") or r.get("error") or "No output returned"
                            ui.label(reply_content).classes("font-mono text-gray-200 break-words whitespace-pre-wrap")

        _render_chat_messages()

        # Direct Chat Input Box
        with ui.row().classes("w-full items-center gap-2 mt-2"):
            msg_input = ui.input(
                placeholder=f"Send instruction or talk to {task.id}...",
            ).props("outlined dense dark").classes("flex-1 text-xs").on("keydown.enter", lambda: _send_msg())

            def _send_msg() -> None:
                txt = msg_input.value.strip()
                if not txt:
                    return
                msg_input.value = ""
                # Update task prompt and trigger run on-demand
                task.prompt = txt
                scheduler.save_schedules()
                ui.notify(f"💬 Sent to {task.id}, agent is running...", type="info", position="top-right")
                scheduler.trigger_now(task.id)
                time.sleep(0.5)
                _render_chat_messages()
                on_update()

            ui.button("Send", icon="send", on_click=_send_msg).props("unelevated dense no-caps").classes("bg-sky-600 hover:bg-sky-500 text-white font-medium px-4 rounded-lg text-xs")

    dialog.open()


def _open_run_logs_modal(task_id: str) -> None:
    """Open modal dialog displaying past run logs for a task."""
    dialog = ui.dialog()
    with dialog, ui.card().classes("w-[90vw] max-w-[700px] bg-[#18181c] border border-gray-800 p-4 rounded-xl flex flex-col gap-3 text-white"):
        with ui.row().classes("w-full justify-between items-center border-b border-gray-800 pb-2"):
            ui.label(f"📜 Execution Run Logs — {task_id}").classes("text-sm font-bold text-white font-mono")
            ui.button(icon="close", on_click=dialog.close).props("flat dense round").classes("text-gray-400")

        runs = load_task_run_logs(task_id)

        if not runs:
            ui.label("No past run logs found for this task.").classes("text-xs text-gray-500 py-4")
        else:
            with ui.column().classes("w-full max-h-[450px] overflow-y-auto gap-2 pr-1"):
                for r in runs:
                    status_color = "text-emerald-400" if r.get("status") == "succeeded" else "text-rose-400"
                    t_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(r.get("started_at", 0)))

                    with ui.card().classes("w-full bg-[#222228] border border-gray-800 p-3 rounded-lg flex flex-col gap-1 text-xs"):
                        with ui.row().classes("w-full justify-between items-center"):
                            ui.label(f"ID: {r.get('run_id')}").classes("font-mono font-bold text-gray-300")
                            ui.label(f"{r.get('status', '').upper()} ({r.get('duration_seconds', 0)}s)").classes(f"font-bold {status_color}")

                        ui.label(f"Time: {t_str}").classes("text-[10px] text-gray-400 font-mono")

                        if r.get("result"):
                            ui.label("Result:").classes("text-[11px] font-bold text-gray-300 mt-1")
                            ui.label(r["result"]).classes("text-[11px] text-gray-300 bg-[#16161a] p-2 rounded font-mono break-all")

                        if r.get("error"):
                            ui.label("Error:").classes("text-[11px] font-bold text-rose-400 mt-1")
                            ui.label(r["error"]).classes("text-[11px] text-rose-300 bg-rose-950/50 p-2 rounded font-mono break-all border border-rose-900")

    dialog.open()


def _create_schedule_modal(scheduler: Scheduler, on_saved: Callable[[], None]) -> None:
    """Open dialog to create a new scheduled task."""
    dialog = ui.dialog()
    with dialog, ui.card().classes("w-[90vw] max-w-[500px] bg-[#18181c] border border-gray-800 p-5 rounded-xl flex flex-col gap-3 text-white"):
        ui.label("➕ Create Scheduled Subagent Task").classes("text-sm font-bold text-white border-b border-gray-800 pb-2")

        type_select = ui.select(
            options={"cron": "Recurring Cron Job", "timer": "One-Shot Timer"},
            value="cron",
            label="Schedule Type",
        ).props("outlined dense dark").classes("w-full")

        cron_input = ui.input(
            label="Cron Expression (5 fields)",
            placeholder="e.g. */5 * * * *",
            value="*/5 * * * *",
        ).props("outlined dense dark").classes("w-full")

        timer_input = ui.input(
            label="Duration in Seconds",
            placeholder="e.g. 60",
            value="60",
        ).props("outlined dense dark").classes("w-full hidden")

        def _on_type_change(e: Any) -> None:
            if e.value == "timer":
                cron_input.classes(add="hidden")
                timer_input.classes(remove="hidden")
            else:
                cron_input.classes(remove="hidden")
                timer_input.classes(add="hidden")

        type_select.on_value_change(_on_type_change)

        prompt_input = ui.textarea(
            label="Subagent Task Prompt",
            placeholder="e.g. Fetch current BTC price online and summarize status",
        ).props("outlined dense dark").classes("w-full")

        with ui.row().classes("w-full justify-end gap-2 mt-2"):
            ui.button("Cancel", on_click=dialog.close).props("flat").classes("text-gray-400")

            def _save() -> None:
                prompt = prompt_input.value.strip()
                if not prompt:
                    ui.notify("Prompt cannot be empty", type="warning", position="top-right")
                    return

                try:
                    if type_select.value == "cron":
                        task_id = scheduler.schedule_cron(cron_input.value.strip(), prompt=prompt)
                    else:
                        task_id = scheduler.schedule_timer(float(timer_input.value.strip()), prompt=prompt)

                    ui.notify(f"✅ Created schedule: {task_id}", type="positive", position="top-right")
                    on_saved()
                    dialog.close()
                except Exception as exc:
                    ui.notify(f"Error: {exc}", type="negative", position="top-right")

            ui.button("Save Schedule", on_click=_save).props("unelevated dense no-caps").classes("bg-violet-600 text-white font-medium px-4 rounded-lg")

    dialog.open()
