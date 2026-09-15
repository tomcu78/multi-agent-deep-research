"""Data models for final structured reports and synthesized publications."""
import re
from typing import List
from pydantic import BaseModel, Field
from src.models.finding import SourceCitation


class ReportSection(BaseModel):
    """A detailed section in the final research report."""
    title: str = Field(description="Section heading (e.g. 'État de l'art')")
    content_markdown: str = Field(
        description="Section body in Markdown with inline numeric citations like [1], [2]"
    )
    key_takeaways: List[str] = Field(
        default_factory=list,
        description="Optional bullet points"
    )
    citations_used: List[int] = Field(
        default_factory=list,
        description="List of SourceCitation IDs referenced in this section"
    )


class ResearchReport(BaseModel):
    """Full deep research report with structured sections and verifiable bibliography."""
    title: str = Field(description="Comprehensive title of the report")
    subtitle: str = Field(default="", description="Subtitle or research thesis statement")
    executive_summary: str = Field(
        description="Executive summary synthesizing findings"
    )
    methodology: str = Field(
        description="Overview of how research was conducted"
    )
    sections: List[ReportSection] = Field(
        description="In-depth analytical sections"
    )
    critical_analysis_and_limitations: str = Field(
        description="Discussion on uncertainties or limitations"
    )
    bibliography: List[SourceCitation] = Field(
        default_factory=list,
        description="Complete list of cited sources mapped to their numerical indices"
    )
    full_markdown: str = Field(
        default="",
        description="Assembled complete markdown document"
    )

    def assemble_markdown(self) -> str:
        """Render the complete structured report into clean editorial Markdown."""
        lines = []
        lines.append(f"# {self.title}")
        if self.subtitle:
            lines.append(f"*{self.subtitle}*\n")
        else:
            lines.append("")

        lines.append("## Synthèse générale")
        lines.append(self.executive_summary)
        lines.append("")

        for idx, sec in enumerate(self.sections, 1):
            clean_title = re.sub(r"^\d+\.\s*", "", sec.title.strip())
            lines.append(f"## {idx}. {clean_title}")
            lines.append(sec.content_markdown)
            lines.append("")

        if self.critical_analysis_and_limitations:
            lines.append("## Analyse critique & Limites")
            lines.append(self.critical_analysis_and_limitations)
            lines.append("")

        lines.append("## Sources & Références")
        if self.bibliography:
            for src in self.bibliography:
                domain_str = f" ({src.domain})" if src.domain else ""
                lines.append(f"- **[{src.id}]** [{src.title}]({src.url}){domain_str}")
        else:
            lines.append("Aucune source publique accessible indexée.")

        self.full_markdown = "\n".join(lines)
        return self.full_markdown
