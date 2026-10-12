#!/usr/bin/env python3
"""Check that a checkpoint was trained from scratch by this codebase.

Usage:
    python scripts/verify_provenance.py checkpoints/model_final.pt [--json]

Walks the provenance record saved inside the checkpoint back to its roots and
reports one of:

    scratch     every root is a FramerAI run that loaded nothing    (exit 0)
    unrecorded  some ancestor was saved before provenance existed   (exit 2)
    foreign     some ancestor names another origin                  (exit 1)

See model/provenance.py for what the record does and does not prove.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from model import provenance

EXIT = {"scratch": 0, "foreign": 1, "unrecorded": 2}


def main():
    parser = argparse.ArgumentParser(description="Verify checkpoint provenance")
    parser.add_argument("checkpoint")
    parser.add_argument("--json", action="store_true", help="print the full report as JSON")
    args = parser.parse_args()

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    report = provenance.verify(payload.get("provenance"))
    report["fingerprint"] = provenance.fingerprint(payload["model_state_dict"])

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"{args.checkpoint}: {report['status']}")
        print(f"  roots {report['roots']}, lineage depth {report['depth']}, "
              f"self-distillation steps {report['teachers']}")
        print(f"  weights sha256 {report['fingerprint']}")
    sys.exit(EXIT[report["status"]])


if __name__ == "__main__":
    main()
