import os
from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak

def create_sample_policy_pdf(output_path: str):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        rightMargin=54,
        leftMargin=54,
        topMargin=54,
        bottomMargin=54
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontSize=20,
        leading=24,
        spaceAfter=14
    )
    section_style = ParagraphStyle(
        'DocSection',
        parent=styles['Heading2'],
        fontSize=14,
        leading=18,
        spaceAfter=10
    )
    body_style = ParagraphStyle(
        'DocBody',
        parent=styles['Normal'],
        fontSize=10.5,
        leading=15,
        spaceAfter=10
    )

    story = []

    # PAGE 1: Remote Work Policy
    story.append(Paragraph("Acme Global Technologies - Corporate Policy Handbook", title_style))
    story.append(Paragraph("Section 1: Remote Work & Eligibility Guidelines", section_style))
    story.append(Paragraph(
        "1.1 Eligibility: All full-time employees must have completed their mandatory 90-day probationary period "
        "before requesting permanent or hybrid remote work arrangements. Contractor eligibility is subject to individual client statements of work.",
        body_style
    ))
    story.append(Paragraph(
        "1.2 Core Collaboration Hours: Regardless of local time zones, all remote team members must be reachable and available "
        "for synchronous team meetings during core collaboration hours from 10:00 AM to 3:00 PM Eastern Time, Monday through Thursday.",
        body_style
    ))
    story.append(Paragraph(
        "1.3 Home Office Connectivity: A high-speed broadband internet connection with a minimum bandwidth of 50 Mbps download "
        "and 10 Mbps upload speed is strictly required. Acme Global provides a one-time remote home office stipend of $500 for ergonomic accessories.",
        body_style
    ))
    story.append(PageBreak())

    # PAGE 2: Travel & Meal Expense Reimbursements
    story.append(Paragraph("Acme Global Technologies - Corporate Policy Handbook", title_style))
    story.append(Paragraph("Section 2: Business Travel & Meal Reimbursements", section_style))
    story.append(Paragraph(
        "2.1 Meal Per Diem Limits: Daily meal expense reimbursements are strictly capped at $75 per day, inclusive of applicable taxes "
        "and customary gratuity (standard breakdown: $20 for breakfast, $25 for lunch, and $30 for dinner). Itemized receipts are required for all claims.",
        body_style
    ))
    story.append(Paragraph(
        "2.2 Non-Reimbursable Purchases: Alcohol purchases, minibar refreshments, movie rentals, and optional room service fees are "
        "strictly non-reimbursable under corporate travel guidelines. Personal travel extensions must be billed to personal payment cards.",
        body_style
    ))
    story.append(Paragraph(
        "2.3 Lodging Ceilings: Hotel accommodations must not exceed $250 per night before taxes in standard metropolitan areas, or "
        "$350 per night in designated high-cost cities (such as New York, San Francisco, and London). Bookings must be routed through the Navan portal.",
        body_style
    ))
    story.append(PageBreak())

    # PAGE 3: Annual Paid Time Off (PTO) & Parental Leave
    story.append(Paragraph("Acme Global Technologies - Corporate Policy Handbook", title_style))
    story.append(Paragraph("Section 3: Paid Time Off (PTO) & Family Leave", section_style))
    story.append(Paragraph(
        "3.1 PTO Accrual: Full-time employees accrue 18 days of Paid Time Off (PTO) per calendar year, accrued bi-weekly on a proportional basis. "
        "PTO requests exceeding three consecutive business days must be submitted to direct supervisors at least two weeks in advance.",
        body_style
    ))
    story.append(Paragraph(
        "3.2 Rollover Policy: Employees may rollover a maximum of 5 unused PTO days into the following calendar year. Any additional unused "
        "days exceeding the 5-day ceiling will forfeit on March 31 of the new calendar year without cash payout.",
        body_style
    ))
    story.append(Paragraph(
        "3.3 Parental Leave: Primary caregivers are entitled to 16 weeks of fully paid parental leave following the birth, adoption, or foster placement "
        "of a child. Secondary caregivers are entitled to 6 weeks of fully paid leave, to be taken within the first 12 months.",
        body_style
    ))
    story.append(PageBreak())

    # PAGE 4: Equipment Security & Lost Hardware Procedures
    story.append(Paragraph("Acme Global Technologies - Corporate Policy Handbook", title_style))
    story.append(Paragraph("Section 4: IT Security & Equipment Protection", section_style))
    story.append(Paragraph(
        "4.1 Lost Hardware Incident Reporting: Lost or stolen corporate laptops, tablets, or test phones must be reported to the IT Security Helpdesk "
        "within 2 hours of discovery. An immediate remote cryptographic wipe will be initiated via Mobile Device Management (MDM).",
        body_style
    ))
    story.append(Paragraph(
        "4.2 Authentication Protocols: Multi-Factor Authentication (MFA) with FIDO2 hardware security keys or approved authenticator apps "
        "is mandatory across all single sign-on (SSO) internal portals and cloud infrastructure. SMS-based 2FA is deprecated and disallowed.",
        body_style
    ))
    story.append(Paragraph(
        "4.3 Prohibited Storage Devices: The attachment of personal USB flash drives, external hard drives, or syncing to unapproved consumer cloud drives "
        "(e.g., personal Google Drive or Dropbox) is blocked by endpoint DLP software and constitutes a severe security violation.",
        body_style
    ))

    doc.build(story)
    print(f"Sample PDF successfully created at: {output_path}")

if __name__ == "__main__":
    create_sample_policy_pdf("data/sample_docs/sample_policy.pdf")
