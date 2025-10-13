#!/usr/bin/env python3
"""
ACE Review Queue CLI

Command-line interface for managing insight review queue.
Supports listing pending items, approving, and rejecting low-confidence insights.

Based on tasks.md T062.

Usage:
    python scripts/ace_review.py list [--domain DOMAIN] [--limit N]
    python scripts/ace_review.py approve <insight_id> [--reviewer REVIEWER] [--notes NOTES]
    python scripts/ace_review.py reject <insight_id> [--reviewer REVIEWER] [--notes NOTES]
    python scripts/ace_review.py stats [--domain DOMAIN]
"""

import sys
import argparse
from typing import Optional
from datetime import datetime
from tabulate import tabulate

from ace.ops.review_service import create_review_service
from ace.utils.database import get_session


def cmd_list(
    domain_id: Optional[str] = None,
    limit: Optional[int] = None
) -> None:
    """
    List pending review items.

    T062: CLI command `ace review list`.

    Args:
        domain_id: Optional filter by domain
        limit: Optional limit on number of items
    """
    with get_session() as session:
        review_service = create_review_service(session)
        items = review_service.list_pending(domain_id=domain_id, limit=limit)

        if not items:
            print("No pending review items.")
            return

        # Format as table
        table_data = []
        for item in items:
            # Truncate content for display
            content_display = (
                item.content[:60] + "..." if len(item.content) > 60 else item.content
            )

            table_data.append([
                item.id[:8],  # Short ID
                item.domain_id,
                item.section,
                f"{item.confidence:.2f}",
                content_display,
                item.source_task_id[:8],
                item.created_at.strftime("%Y-%m-%d %H:%M")
            ])

        headers = ["ID", "Domain", "Section", "Conf", "Content", "Task", "Created"]
        print(tabulate(table_data, headers=headers, tablefmt="simple"))
        print(f"\nTotal pending: {len(items)}")


def cmd_approve(
    item_id: str,
    reviewer_id: Optional[str] = None,
    review_notes: Optional[str] = None
) -> None:
    """
    Approve review item and promote to shadow stage.

    T062: CLI command `ace review approve <insight_id>`.

    Args:
        item_id: Review queue item ID
        reviewer_id: Optional identifier of reviewer
        review_notes: Optional notes about the decision
    """
    with get_session() as session:
        review_service = create_review_service(session)

        # Get item details for display
        item = review_service.review_repo.get_by_id(item_id)
        if not item:
            print(f"Error: Review item '{item_id}' not found.")
            sys.exit(1)

        print(f"Approving review item: {item_id}")
        print(f"Content: {item.content}")
        print(f"Confidence: {item.confidence:.2f}")
        print(f"Domain: {item.domain_id}")
        print()

        bullet_id = review_service.approve_and_promote(
            item_id=item_id,
            reviewer_id=reviewer_id,
            review_notes=review_notes
        )

        if bullet_id:
            print(f"✓ Approved and promoted to shadow stage.")
            print(f"Created bullet ID: {bullet_id}")
        else:
            print(f"✗ Failed to approve review item.")
            sys.exit(1)


def cmd_reject(
    item_id: str,
    reviewer_id: Optional[str] = None,
    review_notes: Optional[str] = None
) -> None:
    """
    Reject review item and discard.

    T062: CLI command `ace review reject <insight_id>`.

    Args:
        item_id: Review queue item ID
        reviewer_id: Optional identifier of reviewer
        review_notes: Optional notes about the decision
    """
    with get_session() as session:
        review_service = create_review_service(session)

        # Get item details for display
        item = review_service.review_repo.get_by_id(item_id)
        if not item:
            print(f"Error: Review item '{item_id}' not found.")
            sys.exit(1)

        print(f"Rejecting review item: {item_id}")
        print(f"Content: {item.content}")
        print(f"Confidence: {item.confidence:.2f}")
        print(f"Domain: {item.domain_id}")
        print()

        success = review_service.reject(
            item_id=item_id,
            reviewer_id=reviewer_id,
            review_notes=review_notes
        )

        if success:
            print(f"✓ Review item rejected and discarded.")
        else:
            print(f"✗ Failed to reject review item.")
            sys.exit(1)


def cmd_stats(domain_id: Optional[str] = None) -> None:
    """
    Show review queue statistics.

    Args:
        domain_id: Optional filter by domain
    """
    with get_session() as session:
        review_service = create_review_service(session)
        stats = review_service.get_statistics(domain_id=domain_id)

        domain_str = f" for domain '{domain_id}'" if domain_id else ""
        print(f"Review Queue Statistics{domain_str}:")
        print()
        print(f"Total items:    {stats['total']}")
        print(f"Pending:        {stats['pending']}")
        print(f"Approved:       {stats['approved']}")
        print(f"Rejected:       {stats['rejected']}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="ACE Review Queue Management CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all pending reviews
  python scripts/ace_review.py list

  # List pending reviews for specific domain
  python scripts/ace_review.py list --domain arithmetic

  # List first 10 pending reviews
  python scripts/ace_review.py list --limit 10

  # Approve review item
  python scripts/ace_review.py approve abc123 --reviewer john --notes "Valid insight"

  # Reject review item
  python scripts/ace_review.py reject def456 --reviewer jane --notes "Too vague"

  # Show statistics
  python scripts/ace_review.py stats
  python scripts/ace_review.py stats --domain arithmetic
"""
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    subparsers.required = True

    # List command
    list_parser = subparsers.add_parser("list", help="List pending review items")
    list_parser.add_argument(
        "--domain",
        type=str,
        help="Filter by domain ID"
    )
    list_parser.add_argument(
        "--limit",
        type=int,
        help="Limit number of items"
    )

    # Approve command
    approve_parser = subparsers.add_parser("approve", help="Approve review item")
    approve_parser.add_argument(
        "item_id",
        type=str,
        help="Review item ID to approve"
    )
    approve_parser.add_argument(
        "--reviewer",
        type=str,
        help="Reviewer identifier"
    )
    approve_parser.add_argument(
        "--notes",
        type=str,
        help="Review notes"
    )

    # Reject command
    reject_parser = subparsers.add_parser("reject", help="Reject review item")
    reject_parser.add_argument(
        "item_id",
        type=str,
        help="Review item ID to reject"
    )
    reject_parser.add_argument(
        "--reviewer",
        type=str,
        help="Reviewer identifier"
    )
    reject_parser.add_argument(
        "--notes",
        type=str,
        help="Review notes"
    )

    # Stats command
    stats_parser = subparsers.add_parser("stats", help="Show review queue statistics")
    stats_parser.add_argument(
        "--domain",
        type=str,
        help="Filter by domain ID"
    )

    args = parser.parse_args()

    # Execute command
    try:
        if args.command == "list":
            cmd_list(domain_id=args.domain, limit=args.limit)
        elif args.command == "approve":
            cmd_approve(
                item_id=args.item_id,
                reviewer_id=args.reviewer,
                review_notes=args.notes
            )
        elif args.command == "reject":
            cmd_reject(
                item_id=args.item_id,
                reviewer_id=args.reviewer,
                review_notes=args.notes
            )
        elif args.command == "stats":
            cmd_stats(domain_id=args.domain)
    except KeyboardInterrupt:
        print("\nAborted.")
        sys.exit(130)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
