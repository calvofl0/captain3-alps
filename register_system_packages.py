#!/usr/bin/env python3
"""
Register all system site-packages as installed in the current venv.

This creates .dist-info directories for packages that are accessible via
--system-site-packages but not registered in the venv's package database.

Usage:
    .venv/bin/python register_system_packages.py
    # or from within an activated venv:
    python register_system_packages.py
"""

import sys
import sysconfig
from pathlib import Path
import json
import importlib.metadata
import re


def is_in_venv():
    """Check if running inside a virtual environment."""
    return hasattr(sys, 'real_prefix') or (
        hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix
    )


def get_venv_site_packages():
    """Get the site-packages directory for the current venv."""
    return Path(sysconfig.get_path('purelib'))


def normalize_name(name):
    """Normalize package name according to PEP 503."""
    return re.sub(r"[-_.]+", "_", name).lower()


def get_system_packages():
    """Get all packages visible from system site-packages."""
    packages = {}
    venv_prefix = sys.prefix

    for dist in importlib.metadata.distributions():
        # Get the location of this distribution
        if dist._path:
            location = str(dist._path.parent)
            # Only include if it's NOT in the venv itself
            if not location.startswith(venv_prefix):
                packages[dist.name] = {
                    'version': dist.version,
                    'summary': dist.metadata.get('Summary', ''),
                    'location': location
                }

    return packages


def create_dist_info(site_packages, name, version, summary):
    """Create a minimal .dist-info directory for a package."""
    # Normalize the name for the directory according to PEP 427
    # Distribution names must use underscores, not hyphens
    normalized_name = normalize_name(name)
    dist_info_name = f"{normalized_name}-{version}.dist-info"
    dist_info_path = site_packages / dist_info_name

    # Skip if already exists
    if dist_info_path.exists():
        return False

    dist_info_path.mkdir(parents=True, exist_ok=True)

    # Create METADATA (use original name here, not normalized)
    metadata_content = f"""Metadata-Version: 2.1
Name: {name}
Version: {version}
Summary: {summary or 'System-installed package'}
"""
    (dist_info_path / "METADATA").write_text(metadata_content)

    # Create INSTALLER
    (dist_info_path / "INSTALLER").write_text("system\n")

    # Create top_level.txt (use normalized name)
    (dist_info_path / "top_level.txt").write_text(f"{normalized_name}\n")

    # Create RECORD (empty, as we're not tracking files)
    (dist_info_path / "RECORD").write_text("")

    return True


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Register system site-packages in the current venv"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes"
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove previously created system package metadata"
    )
    args = parser.parse_args()

    # Verify we're in a venv
    if not is_in_venv():
        print("Error: This script must be run from within a virtual environment", file=sys.stderr)
        print(f"Current Python: {sys.executable}", file=sys.stderr)
        print(f"Prefix: {sys.prefix}", file=sys.stderr)
        sys.exit(1)

    print(f"Detected venv: {sys.prefix}")

    # Get venv site-packages directory
    site_packages = get_venv_site_packages()
    print(f"Site-packages: {site_packages}")

    # Handle --clean flag
    if args.clean:
        print("\nCleaning previously created system package metadata...")
        cleaned_count = 0
        for dist_info in site_packages.glob("*.dist-info"):
            installer_file = dist_info / "INSTALLER"
            if installer_file.exists() and installer_file.read_text().strip() == "system":
                if args.dry_run:
                    print(f"  Would remove: {dist_info.name}")
                else:
                    import shutil
                    shutil.rmtree(dist_info)
                    print(f"  ✓ Removed: {dist_info.name}")
                    cleaned_count += 1

        if not args.dry_run:
            print(f"\nCleaned {cleaned_count} system package entries")
        else:
            print(f"\nDry run: would clean {cleaned_count} entries")

        if not args.dry_run:
            return

    # Get system packages visible in the venv
    print("\nScanning for system packages...")
    system_packages = get_system_packages()

    if not system_packages:
        print("No system packages found (venv may not have --system-site-packages enabled)")
        return

    print(f"\nFound {len(system_packages)} system packages:")

    created_count = 0
    skipped_count = 0

    for name, info in sorted(system_packages.items()):
        version = info['version']
        summary = info['summary']

        if args.dry_run:
            normalized = normalize_name(name)
            print(f"  Would register: {name} {version} (as {normalized}-{version}.dist-info)")
        else:
            created = create_dist_info(site_packages, name, version, summary)
            if created:
                print(f"  ✓ Registered: {name} {version}")
                created_count += 1
            else:
                print(f"  - Skipped (exists): {name} {version}")
                skipped_count += 1

    if not args.dry_run:
        print(f"\nSummary: {created_count} registered, {skipped_count} skipped")
    else:
        print(f"\nDry run: would register {len(system_packages)} packages")


if __name__ == "__main__":
    main()
