"""
ReUnity Grounding Techniques Module

This module provides structured grounding techniques for use during
dissociative episodes, flashbacks, and high-entropy states. These
techniques are adapted from evidence-based practices and are designed
to be delivered in a trauma-informed manner.

Key Principles:
- User choice and autonomy in technique selection
- Gradual, non-overwhelming approach
- Respect for individual triggers and preferences
- Integration with entropy monitoring

DISCLAIMER: This is not a clinical or treatment tool. It is a theoretical
and support framework only. These techniques are not a substitute for
professional mental health care. If you are in crisis, please contact
a mental health professional or crisis line.

Crisis Resources:
- National Suicide Prevention Lifeline: 988 (US)
- Crisis Text Line: Text HOME to 741741 (US)
- International Association for Suicide Prevention: https://www.iasp.info/resources/Crisis_Centres/

Author: Christopher Ezernack
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class GroundingCategory(Enum):
    """Categories of grounding techniques."""

    SENSORY = "sensory"  # 5-4-3-2-1, temperature, texture
    COGNITIVE = "cognitive"  # Mental exercises, categories
    PHYSICAL = "physical"  # Movement, breathing
    EMOTIONAL = "emotional"  # Safe place visualization, anchoring
    RELATIONAL = "relational"  # Connection-based grounding


class IntensityLevel(Enum):
    """Intensity level of grounding techniques."""

    GENTLE = "gentle"  # Minimal stimulation
    MODERATE = "moderate"  # Standard techniques
    STRONG = "strong"  # More intense grounding


@dataclass
class GroundingTechnique:
    """A grounding technique with instructions."""

    technique_id: str
    name: str
    category: GroundingCategory
    intensity: IntensityLevel
    description: str
    instructions: list[str]
    duration_minutes: float
    contraindications: list[str] = field(default_factory=list)
    adaptations: dict[str, str] = field(default_factory=dict)
    effectiveness_ratings: list[float] = field(default_factory=list)


@dataclass
class GroundingSession:
    """Record of a grounding session."""

    session_id: str
    technique_id: str
    started_at: float
    completed_at: float | None = None
    entropy_before: float = 0.0
    entropy_after: float | None = None
    effectiveness_rating: float | None = None  # User-rated 0-1
    notes: str = ""


class GroundingTechniquesLibrary:
    """
    Library of grounding techniques with personalization.

    This library provides evidence-based grounding techniques that can
    be personalized based on user preferences, triggers, and effectiveness
    history.

    DISCLAIMER: These techniques are supportive tools, not treatment.
    """

    def __init__(self) -> None:
        """Initialize the grounding techniques library."""
        self._techniques: dict[str, GroundingTechnique] = {}
        self._sessions: list[GroundingSession] = []
        self._user_preferences: dict[str, Any] = {}
        self._contraindicated: set[str] = set()

        # Load default techniques
        self._load_default_techniques()

    def _load_default_techniques(self) -> None:
        """Load the default set of grounding techniques."""

        # 5-4-3-2-1 Sensory Grounding
        self._techniques["5-4-3-2-1"] = GroundingTechnique(
            technique_id="5-4-3-2-1",
            name="5-4-3-2-1 Sensory Grounding",
            category=GroundingCategory.SENSORY,
            intensity=IntensityLevel.GENTLE,
            description=(
                "A classic grounding technique that uses all five senses "
                "to anchor you in the present moment."
            ),
            instructions=[
                "Take a slow, deep breath.",
                "Look around and name 5 things you can SEE.",
                "Notice 4 things you can TOUCH or feel.",
                "Listen for 3 things you can HEAR.",
                "Identify 2 things you can SMELL.",
                "Notice 1 thing you can TASTE.",
                "Take another deep breath and notice how you feel.",
            ],
            duration_minutes=3.0,
            adaptations={
                "visual_impairment": "Focus on touch, sound, smell, and taste.",
                "hearing_impairment": "Focus on visual, touch, smell, and taste.",
                "limited_mobility": "Focus on senses accessible to you.",
            },
        )

        # Box Breathing
        self._techniques["box_breathing"] = GroundingTechnique(
            technique_id="box_breathing",
            name="Box Breathing",
            category=GroundingCategory.PHYSICAL,
            intensity=IntensityLevel.GENTLE,
            description=(
                "A structured breathing technique that helps regulate "
                "the nervous system and reduce anxiety."
            ),
            instructions=[
                "Find a comfortable position.",
                "Breathe IN slowly for 4 counts.",
                "HOLD your breath for 4 counts.",
                "Breathe OUT slowly for 4 counts.",
                "HOLD empty for 4 counts.",
                "Repeat this cycle 4-6 times.",
                "Return to normal breathing when ready.",
            ],
            duration_minutes=2.0,
            contraindications=["respiratory_conditions", "panic_during_breath_holds"],
            adaptations={
                "shorter_counts": "Use 3 counts instead of 4 if needed.",
                "skip_holds": "Just focus on slow in and out if holds feel uncomfortable.",
            },
        )

        # Safe Place Visualization
        self._techniques["safe_place"] = GroundingTechnique(
            technique_id="safe_place",
            name="Safe Place Visualization",
            category=GroundingCategory.EMOTIONAL,
            intensity=IntensityLevel.MODERATE,
            description=(
                "Create or recall a mental image of a place where you "
                "feel completely safe and at peace."
            ),
            instructions=[
                "Close your eyes if comfortable, or soften your gaze.",
                "Imagine a place where you feel completely safe.",
                "This can be real or imaginary.",
                "Notice what you see in this place.",
                "What sounds are present?",
                "What does the air feel like?",
                "Allow yourself to feel the safety of this space.",
                "When ready, slowly return to the present.",
            ],
            duration_minutes=5.0,
            contraindications=["difficulty_with_visualization", "intrusive_imagery"],
            adaptations={
                "no_safe_place": "Imagine a protective bubble or shield around you.",
                "eyes_open": "Look at a calming image while imagining.",
            },
        )

        # Cold Water Grounding
        self._techniques["cold_water"] = GroundingTechnique(
            technique_id="cold_water",
            name="Cold Water Grounding",
            category=GroundingCategory.SENSORY,
            intensity=IntensityLevel.STRONG,
            description=(
                "Using cold water to activate the dive reflex and "
                "quickly shift out of a distressed state."
            ),
            instructions=[
                "Get access to cold water (sink, ice, cold drink).",
                "Splash cold water on your face.",
                "Or hold ice cubes in your hands.",
                "Or run cold water over your wrists.",
                "Focus on the sensation of cold.",
                "Notice how your body responds.",
                "Continue until you feel more present.",
            ],
            duration_minutes=2.0,
            contraindications=["heart_conditions", "raynauds", "cold_sensitivity"],
        )

        # Grounding Through Movement
        self._techniques["movement"] = GroundingTechnique(
            technique_id="movement",
            name="Grounding Through Movement",
            category=GroundingCategory.PHYSICAL,
            intensity=IntensityLevel.MODERATE,
            description=(
                "Using physical movement to reconnect with your body "
                "and release tension."
            ),
            instructions=[
                "Stand up if you're able to.",
                "Feel your feet firmly on the ground.",
                "Slowly shift your weight from one foot to the other.",
                "Gently shake out your hands and arms.",
                "Roll your shoulders back and forward.",
                "Take a few steps, noticing each footfall.",
                "Stretch in any way that feels good.",
            ],
            duration_minutes=3.0,
            contraindications=["mobility_limitations"],
            adaptations={
                "seated": "Do gentle movements while seated.",
                "lying_down": "Tense and release muscle groups.",
            },
        )

        # Categories Game
        self._techniques["categories"] = GroundingTechnique(
            technique_id="categories",
            name="Categories Mental Game",
            category=GroundingCategory.COGNITIVE,
            intensity=IntensityLevel.GENTLE,
            description=(
                "A cognitive grounding technique that engages the "
                "thinking brain to interrupt distressing thoughts."
            ),
            instructions=[
                "Choose a category (colors, animals, countries, etc.).",
                "Name items in that category for each letter of the alphabet.",
                "A is for... B is for... and so on.",
                "If you get stuck, skip to the next letter.",
                "Try to get through as many letters as you can.",
                "Notice how your mind shifts focus.",
            ],
            duration_minutes=3.0,
            adaptations={
                "simpler": "Just name 10 items in a category.",
                "harder": "Use more specific categories.",
            },
        )

        # Body Scan
        self._techniques["body_scan"] = GroundingTechnique(
            technique_id="body_scan",
            name="Progressive Body Scan",
            category=GroundingCategory.PHYSICAL,
            intensity=IntensityLevel.GENTLE,
            description=(
                "Systematically bringing awareness to different parts "
                "of your body to increase present-moment awareness."
            ),
            instructions=[
                "Find a comfortable position.",
                "Start by noticing your feet.",
                "What sensations are present there?",
                "Slowly move attention up to your ankles, calves, knees...",
                "Continue through your thighs, hips, belly...",
                "Notice your chest, shoulders, arms, hands...",
                "Finally, notice your neck, face, and head.",
                "Take a moment to feel your whole body.",
            ],
            duration_minutes=5.0,
            contraindications=["body_awareness_triggers"],
            adaptations={
                "partial": "Focus only on hands and feet.",
                "external": "Focus on where your body touches surfaces.",
            },
        )

        # Anchor Object
        self._techniques["anchor_object"] = GroundingTechnique(
            technique_id="anchor_object",
            name="Anchor Object Grounding",
            category=GroundingCategory.SENSORY,
            intensity=IntensityLevel.GENTLE,
            description=(
                "Using a physical object as an anchor to the present "
                "moment and to positive memories or feelings."
            ),
            instructions=[
                "Hold your anchor object (stone, jewelry, fabric, etc.).",
                "Notice its weight in your hand.",
                "Feel its texture with your fingers.",
                "Notice its temperature.",
                "If it has a scent, notice that too.",
                "Remember why this object is meaningful to you.",
                "Let it remind you that you are here, now, and safe.",
            ],
            duration_minutes=2.0,
            adaptations={
                "no_object": "Use any nearby object as a temporary anchor.",
            },
        )

        # Orienting to the Present
        self._techniques["orienting"] = GroundingTechnique(
            technique_id="orienting",
            name="Orienting to the Present",
            category=GroundingCategory.COGNITIVE,
            intensity=IntensityLevel.GENTLE,
            description=(
                "Consciously orienting yourself to the present moment "
                "by stating facts about where and when you are."
            ),
            instructions=[
                "Say out loud or to yourself:",
                "My name is [your name].",
                "I am [your age] years old.",
                "Today is [day and date].",
                "I am in [location].",
                "I am safe right now.",
                "The difficult moment will pass.",
                "Repeat as needed until you feel more present.",
            ],
            duration_minutes=2.0,
        )

        # Butterfly Hug
        self._techniques["butterfly_hug"] = GroundingTechnique(
            technique_id="butterfly_hug",
            name="Butterfly Hug (Self-Soothing)",
            category=GroundingCategory.EMOTIONAL,
            intensity=IntensityLevel.GENTLE,
            description=(
                "A bilateral stimulation technique that can help "
                "calm the nervous system through self-touch."
            ),
            instructions=[
                "Cross your arms over your chest.",
                "Place your hands on your upper arms or shoulders.",
                "Alternately tap your hands, left then right.",
                "Tap at a comfortable, slow rhythm.",
                "Breathe slowly as you tap.",
                "Continue for 1-2 minutes.",
                "Notice any shift in how you feel.",
            ],
            duration_minutes=2.0,
            adaptations={
                "hands_busy": "Alternate tapping your feet instead.",
            },
        )

    def get_technique(self, technique_id: str) -> GroundingTechnique | None:
        """Get a technique by ID."""
        return self._techniques.get(technique_id)

    def list_techniques(
        self,
        category: GroundingCategory | None = None,
        intensity: IntensityLevel | None = None,
        exclude_contraindicated: bool = True,
    ) -> list[GroundingTechnique]:
        """
        List available techniques with optional filtering.

        Args:
            category: Filter by category.
            intensity: Filter by intensity.
            exclude_contraindicated: Exclude user's contraindicated techniques.

        Returns:
            List of matching techniques.
        """
        techniques = list(self._techniques.values())

        if category:
            techniques = [t for t in techniques if t.category == category]

        if intensity:
            techniques = [t for t in techniques if t.intensity == intensity]

        if exclude_contraindicated:
            techniques = [
                t for t in techniques
                if t.technique_id not in self._contraindicated
            ]

        return techniques

    def recommend_technique(
        self,
        entropy_level: float,
        category_preference: GroundingCategory | None = None,
    ) -> GroundingTechnique | None:
        """
        Recommend a technique based on current state.

        Args:
            entropy_level: Current entropy level (0-1).
            category_preference: Preferred category if any.

        Returns:
            Recommended technique or None.
        """
        available = self.list_techniques(
            category=category_preference,
            exclude_contraindicated=True,
        )

        if not available:
            return None

        # Select intensity based on entropy
        if entropy_level < 0.5:
            preferred_intensity = IntensityLevel.GENTLE
        elif entropy_level < 0.75:
            preferred_intensity = IntensityLevel.MODERATE
        else:
            preferred_intensity = IntensityLevel.STRONG

        # Filter by intensity
        intensity_matched = [
            t for t in available
            if t.intensity == preferred_intensity
        ]

        if intensity_matched:
            available = intensity_matched

        # Prioritize techniques with good effectiveness ratings
        rated = [
            t for t in available
            if t.effectiveness_ratings and sum(t.effectiveness_ratings) / len(t.effectiveness_ratings) > 0.6
        ]

        if rated:
            available = rated

        # Return random selection from available
        return random.choice(available) if available else None

    def start_session(
        self,
        technique_id: str,
        entropy_before: float,
    ) -> GroundingSession | None:
        """
        Start a grounding session.

        Args:
            technique_id: ID of the technique to use.
            entropy_before: Entropy level before starting.

        Returns:
            The created session or None if technique not found.
        """
        if technique_id not in self._techniques:
            return None

        session = GroundingSession(
            session_id=f"gs_{int(time.time())}_{technique_id}",
            technique_id=technique_id,
            started_at=time.time(),
            entropy_before=entropy_before,
        )

        self._sessions.append(session)
        return session

    def complete_session(
        self,
        session_id: str,
        entropy_after: float,
        effectiveness_rating: float | None = None,
        notes: str = "",
    ) -> bool:
        """
        Complete a grounding session.

        Args:
            session_id: ID of the session.
            entropy_after: Entropy level after completion.
            effectiveness_rating: User's rating of effectiveness (0-1).
            notes: Any notes about the session.

        Returns:
            True if session was found and completed.
        """
        for session in self._sessions:
            if session.session_id == session_id:
                session.completed_at = time.time()
                session.entropy_after = entropy_after
                session.effectiveness_rating = effectiveness_rating
                session.notes = notes

                # Update technique effectiveness ratings
                if effectiveness_rating is not None:
                    technique = self._techniques.get(session.technique_id)
                    if technique:
                        technique.effectiveness_ratings.append(effectiveness_rating)

                return True

        return False

    def add_contraindication(self, technique_id: str) -> None:
        """Mark a technique as contraindicated for this user."""
        self._contraindicated.add(technique_id)

    def remove_contraindication(self, technique_id: str) -> None:
        """Remove a contraindication."""
        self._contraindicated.discard(technique_id)

    def get_session_history(
        self,
        limit: int = 50,
    ) -> list[GroundingSession]:
        """Get recent session history."""
        return self._sessions[-limit:]

    def get_effectiveness_summary(self) -> dict[str, Any]:
        """
        Get summary of technique effectiveness.

        Returns:
            Dictionary with effectiveness statistics.
        """
        summary = {}

        for technique_id, technique in self._techniques.items():
            if technique.effectiveness_ratings:
                ratings = technique.effectiveness_ratings
                summary[technique_id] = {
                    "name": technique.name,
                    "average_rating": sum(ratings) / len(ratings),
                    "times_used": len(ratings),
                    "category": technique.category.value,
                }

        return summary

    def add_custom_technique(
        self,
        name: str,
        category: GroundingCategory,
        intensity: IntensityLevel,
        description: str,
        instructions: list[str],
        duration_minutes: float = 3.0,
    ) -> GroundingTechnique:
        """
        Add a custom grounding technique.

        Args:
            name: Name of the technique.
            category: Category of the technique.
            intensity: Intensity level.
            description: Description of the technique.
            instructions: Step-by-step instructions.
            duration_minutes: Estimated duration.

        Returns:
            The created technique.
        """
        technique_id = f"custom_{int(time.time())}_{name.lower().replace(' ', '_')}"

        technique = GroundingTechnique(
            technique_id=technique_id,
            name=name,
            category=category,
            intensity=intensity,
            description=description,
            instructions=instructions,
            duration_minutes=duration_minutes,
        )

        self._techniques[technique_id] = technique
        return technique
