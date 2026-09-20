# Feature Specification: Scheduled Headless Subagents

## Overview
This document outlines the architecture and design for transforming scheduled timers and cron tasks into **headless subagent background workers with isolated sessions**, persistent logging, push notifications, and a dedicated management dashboard in Unjess.

---

## 1. Current System Gaps vs. Ideal Design

| Feature | Current Behavior | Proposed Design (Headless Subagents) |
|---|---|---|
| **Session Isolation** | Fires callbacks into the active main chat context | Each scheduled task spawns a dedicated subagent with an isolated conversation session |
| **Delivery & Alerting** | Ephemeral chat messages | Push notification toast + persistent execution log |
| **Persistence** | In-memory timers (lost on app restart) | Disk persistence in `~/.unjess/schedules.json` (survives app restarts) |
| **Visibility & Control** | Chat commands only (`/schedule`) | Full Schedule Dashboard UI (view, pause, edit, delete, run history) |
| **Concurrency** | Shares main agent loop | Independent background worker threads with isolated state |

---

## 2. Core Mental Model
> **"A scheduled task is a headless subagent with its own session, its own output log, and a push notification when it finishes — not a timer that interrupts the main chat."**

---

## 3. Key Components & Implementation Plan

### Phase 1: Disk Persistence & Task Metadata (`~/.unjess/schedules.json`)
- Persist scheduled tasks to disk (`~/.unjess/schedules.json`).
- Automatically reload active cron schedules and pending timers on application launch.
- Track `last_run`, `next_run`, `status` (`idle`, `running`, `succeeded`, `failed`), and `run_count`.

### Phase 2: Headless Subagent Execution per Scheduled Trigger
- When a cron or timer triggers, spawn a dedicated subagent (`task_type="scheduled"`).
- Run the prompt in a clean background subagent context without injecting messages into the main chat window.
- Store run logs in `~/.unjess/schedules/<task_id>/runs/<timestamp>.json`.

### Phase 3: Push Notifications & Toast Alerts
- On completion, emit a desktop toast notification:  
  `✅ Scheduled Task Finished: "BTC Price Check"`  
  *(With a clickable action to open that subagent's run log/session tab)*.

### Phase 4: Schedule Management Dashboard (GUI Modal / Panel)
- Add a **Schedule Dashboard** card in the GUI:
  - List all active/paused schedules.
  - Display Cron expression, last run time, next run time, and status badge.
  - Action buttons: **Run Now**, **Pause/Resume**, **Edit**, **Delete**, and **View Logs**.

---

## 4. Prioritized Execution Checklist
1. [ ] Disk Persistence for Schedules (`~/.unjess/schedules.json`)
2. [ ] Isolated Subagent Execution per Trigger
3. [ ] Persistent Output Logging (`~/.unjess/schedules/<id>/runs/`)
4. [ ] Push Notification Toasts
5. [ ] Schedule Dashboard UI in GUI Settings / Sidebar
