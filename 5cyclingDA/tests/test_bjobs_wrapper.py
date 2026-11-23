"""Tests for the bjobs/bhist wrappers used by the 50-member controller.

The fixture is a real `bhist -l 20776452` output from lililei1 after the
2026-11-23 WRF array attempt: 44 members DONE, and members 22/23/30/32/39/49
EXIT.  The test verifies that the wrapper reconstructs all 50 per-element
states from that history, which the driver's built-in bjobs query cannot do.
"""
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
BIN = HERE.parent / "bin"
FIXTURE = HERE / "fixtures" / "bhist_20776452.txt"
JOB_ID = "20776452"

sys.path.insert(0, str(BIN))
from lsf_state_cache import parse_bhist  # noqa: E402


def write_executable(path, text):
    path.write_text(text)
    path.chmod(0o755)


class BjobsWrapperTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.cache = Path(tempfile.gettempdir()) / (
            "lsf_state_%s_%s.json" % (os.getuid(), JOB_ID))
        if self.cache.exists():
            self.cache.unlink()

        jobs = self.root / "oceanAssim1Run1" / "EAKF" / "jobs" / "10_00_00" / "attempt_01"
        jobs.mkdir(parents=True)
        (jobs / "lsf_job.json").write_text(json.dumps({
            "job_id": JOB_ID,
            "members": list(range(1, 51)),
            "name": "c50_test_10_00_00_wrf1",
            "state": "SUBMITTED",
        }))

        self.mock_bin = self.root / "mock_bin"
        self.mock_bin.mkdir()
        write_executable(self.mock_bin / "bhist",
                         "#!/bin/sh\ncat %s\n" % str(FIXTURE))
        write_executable(self.mock_bin / "bjobs",
                         "#!/bin/sh\necho '12345 PEND'\n")

        self.env = dict(os.environ)
        self.env.update({
            "RUN_ROOT": str(self.root),
            "REAL_BHIST": str(self.mock_bin / "bhist"),
            "REAL_BJOBS": str(self.mock_bin / "bjobs"),
            "PATH": str(self.mock_bin) + os.pathsep + self.env.get("PATH", ""),
        })

    def tearDown(self):
        if self.cache.exists():
            self.cache.unlink()
        self.tmp.cleanup()

    def run_wrapper(self, name, args):
        return subprocess.run(
            [sys.executable, str(BIN / name)] + args,
            env=self.env, capture_output=True, text=True)

    def test_array_states_are_reconstructed_from_bhist(self):
        result = self.run_wrapper(
            "bjobs", ["-a", "-noheader", "-o", "jobid stat", JOB_ID])
        self.assertEqual(result.returncode, 0, result.stderr)
        states = {}
        for line in result.stdout.splitlines():
            fields = line.split()
            if len(fields) == 2:
                states[fields[0]] = fields[1]
        self.assertEqual(len(states), 50)
        from collections import Counter
        counts = Counter(states.values())
        self.assertEqual(counts["DONE"], 44)
        self.assertEqual(counts["EXIT"], 6)
        self.assertEqual(states[JOB_ID + "[1]"], "DONE")
        self.assertEqual(states[JOB_ID + "[22]"], "EXIT")
        self.assertEqual(states[JOB_ID + "[23]"], "EXIT")
        self.assertEqual(states[JOB_ID + "[30]"], "EXIT")
        self.assertEqual(states[JOB_ID + "[32]"], "EXIT")
        self.assertEqual(states[JOB_ID + "[39]"], "EXIT")
        self.assertEqual(states[JOB_ID + "[49]"], "EXIT")
        self.assertEqual(states[JOB_ID + "[50]"], "DONE")

    def test_bhist_fallback_reads_cached_state(self):
        # Populate the cache through the bjobs wrapper.
        self.run_wrapper(
            "bjobs", ["-a", "-noheader", "-o", "jobid stat", JOB_ID])
        done = self.run_wrapper("bhist", ["-l", JOB_ID + "[1]"])
        self.assertIn("Done successfully", done.stdout)
        exited = self.run_wrapper("bhist", ["-l", JOB_ID + "[22]"])
        self.assertIn("Exited", exited.stdout)
        killed = self.run_wrapper("bhist", ["-l", JOB_ID + "[32]"])
        self.assertIn("Exited", killed.stdout)

    def test_non_array_query_passes_through(self):
        result = self.run_wrapper(
            "bjobs", ["-a", "-noheader", "-o", "jobid stat", "12345"])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "12345 PEND")

    def test_parse_bhist_pending_running_and_terminal_states(self):
        text = """\
Job <9[1]>, Job Name <x[1]>
Mon Nov 23 13:31:09: Submitted from host <amd01>, to Queue <mpi>
Job <9[2]>, Job Name <x[2]>
Mon Nov 23 13:31:14: Dispatched 144 Task(s) on Host(s)
Mon Nov 23 13:31:54: Starting (Pid 1);
Mon Nov 23 13:31:56: Running with execution home </share/home/x>
Job <9[3]>, Job Name <x[3]>
Mon Nov 23 13:31:56: Running with execution home </share/home/x>
Mon Nov 23 13:32:15: Exited with exit code 254.
Job <9[4]>, Job Name <x[4]>
Mon Nov 23 13:59:01: Done successfully. The CPU time used is 1.0 seconds;
"""
        self.assertEqual(parse_bhist(text), {
            1: "PEND",
            2: "RUN",
            3: "EXIT",
            4: "DONE",
        })

    def run_with_mocks(self, bhist_text, bjobs_text):
        mock = self.root / "mock_alt"
        mock.mkdir(exist_ok=True)
        write_executable(mock / "bhist",
                         "#!/bin/sh\nprintf '%s\\n' " + shlex.quote(bhist_text))
        write_executable(mock / "bjobs",
                         "#!/bin/sh\nprintf '%s\\n' " + shlex.quote(bjobs_text))
        env = dict(self.env, REAL_BHIST=str(mock / "bhist"),
                   REAL_BJOBS=str(mock / "bjobs"))
        return subprocess.run(
            [sys.executable, str(BIN / "bjobs"),
             "-a", "-noheader", "-o", "jobid stat", JOB_ID],
            env=env, capture_output=True, text=True)

    @staticmethod
    def parse_states(text):
        states = {}
        for line in text.splitlines():
            fields = line.split()
            if len(fields) == 2:
                states[fields[0]] = fields[1]
        return states

    def test_purged_history_uses_filesystem_markers(self):
        experiment = self.root / "oceanAssim1Run1" / "EAKF"
        (experiment / "run_wrf" / "10_00_00" / "001").mkdir(parents=True)
        (experiment / "run_wrf" / "10_00_00" / "001" / "wrf_done").write_text("id")
        (experiment / "wrfout" / "10_00_00" / "002").mkdir(parents=True)
        (experiment / "wrfout" / "10_00_00" / "002" / "member_done.json").write_text("{}")

        result = self.run_with_mocks("No matching job found",
                                     "Job <20776452> is not found")
        self.assertEqual(result.returncode, 0, result.stderr)
        states = self.parse_states(result.stdout)
        self.assertEqual(len(states), 50)
        self.assertEqual(states[JOB_ID + "[1]"], "DONE")
        self.assertEqual(states[JOB_ID + "[2]"], "DONE")
        self.assertEqual(states[JOB_ID + "[3]"], "EXIT")
        from collections import Counter
        counts = Counter(states.values())
        self.assertEqual(counts["DONE"], 2)
        self.assertEqual(counts["EXIT"], 48)

    def test_active_array_without_history_is_pending(self):
        result = self.run_with_mocks("No matching job found", "20776452 RUN")
        self.assertEqual(result.returncode, 0, result.stderr)
        states = self.parse_states(result.stdout)
        self.assertEqual(len(states), 50)
        self.assertEqual(set(states.values()), {"PEND"})


if __name__ == "__main__":
    unittest.main()
