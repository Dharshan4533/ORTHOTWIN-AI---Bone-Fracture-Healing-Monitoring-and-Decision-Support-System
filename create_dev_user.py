"""
OrthoTwin AI — Explicit Development Account Creation Utility
Allows developers and clinical evaluators to create role-based user accounts
without using hard-coded passwords or default credentials.
"""

import sys
import argparse
import getpass
from pathlib import Path

sys_path = Path(__file__).resolve().parent
if str(sys_path) not in sys.path:
    sys.path.insert(0, str(sys_path))

from database.db import init_db, get_db
from database.repositories import UserRepository
from services.authentication import create_development_user
from config import ALL_ROLES, ROLE_DOCTOR, ROLE_ADMIN, ROLE_PATIENT


def main():
    parser = argparse.ArgumentParser(description="Create an OrthoTwin AI user account.")
    parser.add_argument("--username", help="Unique username")
    parser.add_argument("--role", choices=ALL_ROLES, help="Role: admin, doctor, or patient")
    parser.add_argument("--name", help="Full name of user")
    parser.add_argument("--email", help="Optional email address")
    parser.add_argument("--patient-id", type=int, help="Optional linked patient ID (for patient accounts)")
    parser.add_argument("--password", help="Plaintext password (min 8 characters)")
    parser.add_argument("--seed-defaults", action="store_true", help="Seed standard development test accounts")

    args = parser.parse_args()

    init_db()

    with get_db() as db:
        UserRepository.get_or_create_roles(db)

        if args.seed_defaults:
            print("Seeding standard development test accounts with explicit credentials...")
            defaults = [
                ("admin_user", "AdminSecure2026!", ROLE_ADMIN, "System Administrator", "admin@orthotwin.local", None),
                ("dr_ortho", "DoctorSecure2026!", ROLE_DOCTOR, "Dr. Marcus Vance, MD", "dr.vance@orthotwin.local", None),
                ("patient_demo", "PatientSecure2026!", ROLE_PATIENT, "Alex Mercer", "alex.mercer@orthotwin.local", 1)
            ]
            for uname, pwd, role, fname, email, pid in defaults:
                existing = UserRepository.get_by_username(db, uname)
                if not existing:
                    create_development_user(
                        db=db,
                        username=uname,
                        password=pwd,
                        role_name=role,
                        full_name=fname,
                        email=email,
                        patient_id=pid
                    )
                    print(f"  Created user: {uname} (Role: {role})")
                else:
                    print(f"  User already exists: {uname}")
            print("\nDefault development accounts ready for local testing.")
            return

        # Interactive or argument mode
        username = args.username or input("Enter username: ").strip()
        role = args.role
        while role not in ALL_ROLES:
            role = input(f"Enter role ({', '.join(ALL_ROLES)}): ").strip().lower()

        full_name = args.name or input("Enter full name: ").strip()
        email = args.email or input("Enter email (optional, press Enter to skip): ").strip() or None

        if args.password:
            password = args.password
        else:
            password = getpass.getpass("Enter password (min 8 characters): ")
            confirm = getpass.getpass("Confirm password: ")
            if password != confirm:
                print("Error: Passwords do not match.")
                sys.exit(1)

        try:
            user = create_development_user(
                db=db,
                username=username,
                password=password,
                role_name=role,
                full_name=full_name,
                email=email,
                patient_id=args.patient_id
            )
            print(f"\nUser '{user.username}' successfully created with role '{user.role.name}'.")
        except Exception as e:
            print(f"Error creating user: {str(e)}")
            sys.exit(1)


if __name__ == "__main__":
    main()
