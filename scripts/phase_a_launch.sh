#!/bin/bash
# Boot-time launcher for the Phase A milestone runner (round-2 revision).
# Intended for a user crontab @reboot line, e.g.:
#   @reboot /bin/bash "/home/abrar/Desktop/AI RPM Papers/Cardiac Arrest Detection/scripts/phase_a_launch.sh"
# The runner re-launches dead embedding tags itself and resumes milestones
# from cache/phase_a_state.json, so this is safe to run on every boot.
cd "/home/abrar/Desktop/AI RPM Papers/Cardiac Arrest Detection" || exit 1
source "$HOME/anaconda3/etc/profile.d/conda.sh" && conda activate pyprime || exit 1
setsid python -u scripts/phase_a_runner.py >> logs/phase_a.log 2>&1 < /dev/null &
