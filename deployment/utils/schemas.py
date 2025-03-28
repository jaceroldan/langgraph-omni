from typing import Optional
from pydantic import Field, BaseModel

from graphs.input_handling import InputState


# Tool Schemas
class Project(BaseModel):
    """
        This is the schema format for a project.
    """
    title: Optional[str] = Field("None", description="Title of the project. (eg. The Residences at Greenbelt)")
    project_type: Optional[str] = Field("None", description="Type of project. (eg. Residential - Condominium)")
    description: Optional[str] = Field(
        "None",
        description="Description of the project itself. This can include the location, amenities, and other details."
    )
    location: Optional[str] = Field("None", description="Location of the project. (eg. Legazpi Village, Makati)")
    funding_goal: Optional[str] = Field("None", description="Amount needed to complete the project. (eg. PHP 10M)")
    available_shares: Optional[str] = Field(
        "None", description="Shares of the project available for investment. (eg. 500,000 shares)")
    minimum_viable_fund: Optional[str] = Field(
        "None", description="Minimum amount needed to proceed. (eg. PHP 5M - PHP 10M)")
    funding_date_completion: Optional[str] = Field("None", description="Date of completion for funding. (eg. 2023)")
    key_milestone_dates: list[str] = Field(
        description=("List of important date milestones that investors must take note of."
                     "(eg. Groundbreaking - March 2025, Structure Completion - June 2026)"),
        default_factory=list
    )
    financial_documents: list[str] = Field(
        description=("List of relevant financial documents to strengthen the proposal. "
                     "(eg. business plans, revenue projections"),
        default_factory=list
    )
    legal_documents: list[str] = Field(
        description=("List of relevant legal documents to ensure compliance with local regulations."
                     "(eg. business registrations, land titles, permits)"),
        default_factory=list
    )


# State Schemas
class ProjectState(InputState):
    """
        Used to transfer project information in-between subgraphs.
        =@> Currently does nothing.
    """
    project_details: Project
