#!/usr/bin/env python3
"""
ReUnity Alter-Aware Subsystem Example

This example demonstrates the Alter-Aware Subsystem (AAS) for supporting
individuals with dissociative identity experiences. It shows how to:
- Register and manage alter profiles
- Track switches between identity states
- Facilitate internal communication
- Maintain separate but connected memory spaces

DISCLAIMER: ReUnity is NOT a clinical or treatment tool. It is a theoretical
and support framework only. This system does not diagnose or treat any
condition. Please work with qualified mental health professionals.

If you are in crisis, please contact:
- National Suicide Prevention Lifeline: 988 (US)
- Crisis Text Line: Text HOME to 741741 (US)
"""

import time
from datetime import datetime

from reunity.alter.alter_aware import (
    AlterAwareSubsystem,
    AlterProfile,
    AlterState,
    CommunicationType,
)
from reunity.memory.continuity_store import (
    RecursiveIdentityMemoryEngine,
    ConsentScope,
    MemoryType,
)
from reunity.memory.timeline_threading import (
    TimelineThreader,
    ThreadType,
    MemoryValence,
)


def main():
    print("=" * 60)
    print("ReUnity Alter-Aware Subsystem Example")
    print("=" * 60)
    print()
    print("DISCLAIMER: This is not a clinical or treatment tool.")
    print("Please work with qualified mental health professionals.")
    print()

    # Initialize components
    alter_subsystem = AlterAwareSubsystem()
    memory_engine = RecursiveIdentityMemoryEngine()
    timeline = TimelineThreader()

    # Register system members
    print("-" * 40)
    print("1. Registering System Members")
    print("-" * 40)

    # Register host
    host_profile = AlterProfile(
        alter_id="",  # Will be generated
        name="Alex",
        pronouns="they/them",
        age_presentation="adult",
        role="host",
        communication_style="analytical",
        preferences={"color": "blue", "music": "classical"},
        boundaries=["no discussion of work stress after 8pm"],
    )
    host_id = alter_subsystem.register_alter(host_profile)
    print(f"Registered Host: Alex ({host_id[:8]}...)")

    # Register protector
    protector_profile = AlterProfile(
        alter_id="",
        name="Marcus",
        pronouns="he/him",
        age_presentation="adult",
        role="protector",
        communication_style="direct",
        preferences={"activity": "exercise"},
        boundaries=["respect when I say no"],
    )
    protector_id = alter_subsystem.register_alter(protector_profile)
    print(f"Registered Protector: Marcus ({protector_id[:8]}...)")

    # Register little
    little_profile = AlterProfile(
        alter_id="",
        name="Sunny",
        pronouns="she/her",
        age_presentation="child (7)",
        role="little",
        communication_style="playful",
        preferences={"toy": "stuffed bunny", "snack": "goldfish crackers"},
        boundaries=["no scary topics", "bedtime by 9pm"],
    )
    little_id = alter_subsystem.register_alter(little_profile)
    print(f"Registered Little: Sunny ({little_id[:8]}...)")

    print()

    # Record a switch
    print("-" * 40)
    print("2. Recording a Switch Event")
    print("-" * 40)

    switch_event = alter_subsystem.record_switch(
        from_alter_ids=[host_id],
        to_alter_ids=[protector_id],
        trigger="stressful phone call",
        smoothness=0.6,
        notes="Switched during difficult conversation with boss",
    )
    print(f"Switch recorded: {switch_event.event_id[:8]}...")
    print(f"  From: Alex -> To: Marcus")
    print(f"  Trigger: {switch_event.trigger}")
    print(f"  Smoothness: {switch_event.smoothness}")
    print()

    # Internal communication
    print("-" * 40)
    print("3. Internal Communication")
    print("-" * 40)

    # Marcus sends a message to Alex
    message1 = alter_subsystem.send_internal_message(
        sender_id=protector_id,
        recipient_ids=[host_id],
        content="I handled the call. You can come back when you're ready.",
        message_type=CommunicationType.DIRECT,
    )
    print(f"Marcus -> Alex: '{message1.content}'")

    # System-wide announcement
    message2 = alter_subsystem.send_internal_message(
        sender_id=host_id,
        recipient_ids=[protector_id, little_id],
        content="Thank you Marcus. Sunny, we're okay now.",
        message_type=CommunicationType.BROADCAST,
    )
    print(f"Alex -> Everyone: '{message2.content}'")

    # Leave a note on the internal bulletin board
    message3 = alter_subsystem.send_internal_message(
        sender_id=host_id,
        recipient_ids=[],  # Empty = bulletin board
        content="Therapy appointment tomorrow at 2pm",
        message_type=CommunicationType.BULLETIN,
    )
    print(f"Bulletin Board: '{message3.content}'")
    print()

    # Memory management with identity states
    print("-" * 40)
    print("4. Identity-Aware Memory Management")
    print("-" * 40)

    # Alex's memory (private)
    alex_memory = memory_engine.add_memory(
        identity="Alex",
        content="Had a productive meeting today. Feeling accomplished.",
        memory_type=MemoryType.EPISODIC,
        consent_scope=ConsentScope.PRIVATE,
        tags=["work", "positive"],
    )
    print(f"Alex's private memory: {alex_memory.id[:8]}...")

    # Marcus's memory (system-shared)
    marcus_memory = memory_engine.add_memory(
        identity="Marcus",
        content="Protected the system during a difficult situation.",
        memory_type=MemoryType.EPISODIC,
        consent_scope=ConsentScope.SYSTEM_SHARED,
        tags=["protection", "coping"],
    )
    print(f"Marcus's shared memory: {marcus_memory.id[:8]}...")

    # Sunny's memory (with age-appropriate content)
    sunny_memory = memory_engine.add_memory(
        identity="Sunny",
        content="Drew a picture of a rainbow today! It made me happy.",
        memory_type=MemoryType.EPISODIC,
        consent_scope=ConsentScope.SYSTEM_SHARED,
        tags=["art", "happy", "child-friendly"],
    )
    print(f"Sunny's shared memory: {sunny_memory.id[:8]}...")
    print()

    # Timeline threading
    print("-" * 40)
    print("5. Timeline Threading Across States")
    print("-" * 40)

    # Add memories to timeline
    tm1 = timeline.add_memory(
        content="Morning routine - Alex fronting",
        valence=MemoryValence.NEUTRAL,
        identity_state="Alex",
        timestamp=time.time() - 7200,
    )

    tm2 = timeline.add_memory(
        content="Stressful call - Marcus took over",
        valence=MemoryValence.NEGATIVE,
        identity_state="Marcus",
        timestamp=time.time() - 3600,
    )

    tm3 = timeline.add_memory(
        content="Calming down - Sunny wanted to color",
        valence=MemoryValence.POSITIVE,
        identity_state="Sunny",
        timestamp=time.time() - 1800,
    )

    tm4 = timeline.add_memory(
        content="Evening reflection - Alex back",
        valence=MemoryValence.NEUTRAL,
        identity_state="Alex",
        timestamp=time.time(),
    )

    # Create threads
    alex_thread = timeline.create_thread(
        name="Alex's Day",
        thread_type=ThreadType.IDENTITY,
        memory_ids=[tm1.memory_id, tm4.memory_id],
        description="Memories from Alex's perspective",
    )
    print(f"Created thread: {alex_thread.name}")

    # Check for gaps/switches
    gaps = timeline.get_timeline_gaps()
    switches = [g for g in gaps if g.gap_type == "identity_switch"]
    print(f"Detected {len(switches)} identity switches in timeline")
    print()

    # System report
    print("-" * 40)
    print("6. System Functioning Report")
    print("-" * 40)

    report = alter_subsystem.generate_system_report()
    print(f"Total Alters: {report['total_alters']}")
    print(f"Recent Switches: {report['recent_switch_count']}")
    print(f"Unread Messages: {report['unread_messages']}")
    print(f"Communication Health: {report['communication_health']}")
    print()

    # Export timeline
    print("-" * 40)
    print("7. Timeline Export")
    print("-" * 40)

    export = timeline.export_timeline()
    print(f"Total Memories: {export['statistics']['total_memories']}")
    print(f"Total Threads: {export['statistics']['total_threads']}")
    print(f"Total Gaps: {export['statistics']['total_gaps']}")
    print()

    print("=" * 60)
    print("Example Complete")
    print("=" * 60)
    print()
    print("The Alter-Aware Subsystem supports system functioning by:")
    print("- Recognizing and validating all system members")
    print("- Facilitating healthy internal communication")
    print("- Maintaining appropriate memory boundaries")
    print("- Tracking switches and system patterns")
    print()
    print("Remember: Work with qualified professionals who understand")
    print("dissociative experiences for proper support and treatment.")


if __name__ == "__main__":
    main()
