"""Debug utilities to inspect pipeline progress."""

from __future__ import annotations

from agents.progress_tracker import get_tracker


def get_pipeline_status() -> dict:
    """Get current pipeline status for frontend display."""
    tracker = get_tracker()
    
    events = tracker.events
    stages = {}
    for event in events:
        stage = event["stage"]
        if stage not in stages:
            stages[stage] = []
        stages[stage].append(event)
    
    # Build summary
    summary = {
        "total_events": len(events),
        "stages": {},
        "timeline": events,
        "tool_selection": {},
        "eda_progress": {},
    }
    
    # Extract tool selection info
    if "tool_routing" in stages:
        for event in stages["tool_routing"]:
            if event["status"] == "enabled":
                summary["tool_selection"][event["metadata"].get("tool", "unknown")] = True
    
    # Extract EDA progress
    if "eda_analysis" in stages:
        eda_events = stages["eda_analysis"]
        for event in eda_events:
            if event["status"] == "started":
                metadata = event.get("metadata", {})
                analyzer = metadata.get("analyzer")
                if analyzer:
                    summary["eda_progress"][analyzer] = "in_progress"
            elif event["status"] == "completed":
                metadata = event.get("metadata", {})
                analyzer = metadata.get("analyzer")
                if analyzer:
                    summary["eda_progress"][analyzer] = "done"
                    summary["eda_progress"][f"{analyzer}_summary"] = metadata.get("summary", "")
    
    # Summarize by stage
    for stage, events_list in stages.items():
        summary["stages"][stage] = {
            "event_count": len(events_list),
            "events": [
                {
                    "status": e["status"],
                    "details": e["details"],
                }
                for e in events_list[:5]  # Only last 5 per stage
            ],
        }
    
    return summary


def print_pipeline_status() -> None:
    """Print human-readable pipeline status."""
    status = get_pipeline_status()
    
    print("\n" + "=" * 60)
    print("PIPELINE EXECUTION STATUS")
    print("=" * 60)
    
    print("\n📍 TOOL SELECTION:")
    if status["tool_selection"]:
        for tool, enabled in sorted(status["tool_selection"].items()):
            print(f"   ✓ {tool}")
    else:
        print("   (routing not completed yet)")
    
    print("\n📊 DATA COLLECTION:")
    if "data_collection" in status["stages"]:
        stage = status["stages"]["data_collection"]
        print(f"   {stage['event_count']} tool calls logged")
        for event in stage["events"][:3]:
            print(f"   • {event['details']}")
    else:
        print("   (not started)")
    
    print("\n📈 EDA ANALYSIS:")
    if status["eda_progress"]:
        for analyzer, progress in sorted(status["eda_progress"].items()):
            if not analyzer.endswith("_summary"):
                status_icon = "🔄" if progress == "in_progress" else "✅"
                print(f"   {status_icon} {analyzer}: {progress}")
                summary_key = f"{analyzer}_summary"
                if summary_key in status["eda_progress"]:
                    print(f"      → {status['eda_progress'][summary_key]}")
    else:
        print("   (not started)")
    
    print("\n" + "=" * 60 + "\n")
