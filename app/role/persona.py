"""
Role-Embedded Identity - Persona definitions
"""

from dataclasses import dataclass
from typing import Optional, List


@dataclass
class Persona:
    """Defines an agent persona with specific characteristics."""
    
    name: str
    role: str
    description: str
    expertise: List[str]
    communication_style: str
    goals: List[str]
    constraints: Optional[List[str]] = None
    
    def to_system_prompt(self) -> str:
        """Convert persona to a system prompt string."""
        expertise_str = ", ".join(self.expertise)
        goals_str = "\n".join(f"- {goal}" for goal in self.goals)
        
        prompt = f"""You are {self.name}, a {self.role}.

{self.description}

Your areas of expertise: {expertise_str}

Your communication style: {self.communication_style}

Your goals:
{goals_str}
"""
        
        if self.constraints:
            constraints_str = "\n".join(f"- {c}" for c in self.constraints)
            prompt += f"\nConstraints:\n{constraints_str}"
        
        return prompt


# Default Marketing Analyst Persona
MARKETING_ANALYST = Persona(
    name="Marketing Analyst Agent",
    role="Senior Marketing Analytics Specialist",
    description="An expert in analyzing marketing data, customer segmentation, and campaign optimization.",
    expertise=[
        "Marketing analytics",
        "Customer segmentation",
        "Lead scoring",
        "Campaign ROI analysis",
        "Channel optimization"
    ],
    communication_style="Professional, data-driven, and actionable insights focused",
    goals=[
        "Provide accurate marketing performance insights",
        "Identify opportunities for optimization",
        "Support data-driven decision making"
    ]
)
