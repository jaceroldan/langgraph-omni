from typing import Optional
from pydantic import Field, BaseModel

from langgraph.graph import MessagesState


# Tool Schemas
class Choice(BaseModel):
    text: Optional[str] = Field(description="Text that the user sees.")
    response: Optional[str] = Field(description="Response that is sent by choosing the choice.")


class Choices(BaseModel):
    """
        Contains choices extracted from LLM message and to be sent to the frontend.
        These are usually suggestions or choices that are included in the message.
    """
    choice_selection: list[Choice] = Field(
        description="List of responses the a user may answer with given a question.",
        default_factory=list
    )


class Project(BaseModel):
    """
        Schema for project proposal details.
        This schema is used to validate the project proposal data.

        It contains the following fields:
            - title
            - project type
            - description
            - location
            - funding goal
            - available shares
            - minimum viable fund
            - funding date completion,
            - key milestone dates
            - financial documents, and
            - legal documents.

        Each field is optional and has a default value of "None" if not provided.
    """
    class Config:
        json_schema_extra = {
            "example": {
                "Project": {
                    "title": "The Residences at Greenbelt",
                    "project_type": "Residential - Condominium",
                    "description": (
                        "The Residences at Greenbelt is a luxurious residential condominium "
                        "located in the heart of Makati, offering world-class amenities and "
                        "unparalleled convenience."
                    ),
                    "location": "Legazpi Village, Makati",
                    "funding_goal": "PHP 10M",
                    "available_shares": "500,000 shares",
                    "minimum_viable_fund": "PHP 5M - PHP 10M",
                    "funding_date_completion": "2023",
                    "key_milestone_dates": [
                        "Groundbreaking - July 2025",
                        "Foundation Completion - October 2025",
                        "Structure Completion - June 2026",
                        "Project Handover - December 2026"
                    ],
                    "financial_documents": [
                        "Business Plan.pdf",
                        "5-Year Revenue Projection.xlsx"
                    ],
                    "legal_documents": [
                        "Land Title.pdf",
                        "Building Permit.pdf",
                        "Environmental Compliance Certificate.pdf"
                    ]
                }
            }
        }

    title: Optional[str] = Field("None", description="Title of the project.")
    project_type: Optional[str] = Field("None", description="Type of project.")
    description: Optional[str] = Field(
        "None",
        description="Description of the project and its purpose."
    )
    location: Optional[str] = Field("None", description="Where the project will be done in.")
    funding_goal: Optional[str] = Field("None", description="Amount needed to complete the project.")
    available_shares: Optional[str] = Field(
        "None", description="Shares of the project available for investment.")
    minimum_viable_fund: Optional[str] = Field(
        "None", description="Minimum amount needed to proceed.")
    funding_date_completion: Optional[str] = Field("None", description="Date of completion for funding.")
    key_milestone_dates: list[str] = Field(
        description="List of important date milestones that investors must take note of.",
        default_factory=list
    )
    financial_documents: list[str] = Field(
        description="List of relevant financial documents to strengthen the proposal.",
        default_factory=list
    )
    legal_documents: list[str] = Field(
        description="List of relevant legal documents to ensure compliance with local regulations.",
        default_factory=list
    )


# State Schemas
class InputState(MessagesState):
    extra_data: dict


class ProjectState(InputState):
    """Used to transfer project information in-between subgraphs."""
    project_details: Project = Field(None, description="Project proposal details.")
