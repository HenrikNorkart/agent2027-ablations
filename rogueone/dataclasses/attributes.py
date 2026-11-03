from dataclasses import dataclass
from typing import Literal

Status = Literal["Active", "Pruned"]


@dataclass
class AttributeExplanation:
    """Explanation and metadata for an extracted feature/attribute.
    
    Stores the attribute name, description, pandas command for calculation,
    when it was added, and current processing status.
    
    Attributes:
        name (str): Name of the attribute.
        argumentation (str): Domain reasoning for why this attribute is relevant.
        command (str): Pandas command used to calculate the attribute.
        added (str | int, optional): Iteration when attribute was added.
        status (Status): Current status ("Active" or "Pruned").
    """
    name: str
    argumentation: str
    command: str
    added: str | int | None = None
    status: Status = "Active"

    @classmethod
    def from_dict(cls, data: dict) -> "AttributeExplanation":
        """Create an AttributeExplanation from a dictionary.
        
        Args:
            data (dict): Dictionary with keys: name, description, pd_command, added (optional), status (optional).
            
        Returns:
            AttributeExplanation: Instantiated object.
        """
        return cls(
            name=data["name"],
            argumentation=data["description"],
            command=data["pd_command"],
            added=data.get("added"),
            status=data.get("status", "Active"),
        )

    @property
    def as_dict(self) -> dict:
        return {
            "Attribute": self.name,
            "Description": self.argumentation,
            "Pandas Command": self.command,
            "Status": self.status,
            "Added": self.added,
        }

    def __repr__(self):
        return f"Description: {self.argumentation} - Pandas Command: {self.command}"
