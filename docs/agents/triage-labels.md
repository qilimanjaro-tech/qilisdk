# Triage Labels

The skills speak in terms of five canonical triage roles. This file maps those roles to the actual label strings used in this repo's issue tracker.

| Label in mattpocock/skills | Label in our tracker | Meaning                                  |
| -------------------------- | -------------------- | ---------------------------------------- |
| `needs-triage`             | `needs-triage`       | Maintainer needs to evaluate this issue  |
| `needs-info`               | `question`           | Waiting on reporter for more information |
| `ready-for-agent`          | `ready-for-agent`    | Fully specified, ready for an AFK agent  |
| `ready-for-human`          | `ready-for-human`    | Requires human implementation            |
| `wontfix`                  | `wontfix`            | Will not be actioned                     |

When a skill mentions a role (e.g. "apply the AFK-ready triage label"), use the corresponding label string from this table.

Edit the "Label in our tracker" column to match whatever vocabulary you actually use.

`question` and `wontfix` already exist on GitHub; `question` carries GitHub's default description, "Further information is requested". `needs-triage`, `ready-for-agent` and `ready-for-human` do not exist yet. TODO(human): create them, or map these roles onto existing labels. The "Touches ..." area labels (`core`, `digital`, `qilisim`, ...) say which part of the code an issue touches, and `easy-review` and `priority-review` order the review queue; none of them is a triage state, so they stay as they are.
