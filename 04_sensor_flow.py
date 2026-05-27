"""
ONI Data Sensor Flow
====================
Checks if nino_ml.csv has been updated on GitHub.
If new data is detected, publishes an ArgoEvent to trigger downstream flows.

Mirrors Module 06 from the ml-end-to-end course:
  - Runs on a schedule (daily)
  - Compares current state to previous run's state (self-referencing)
  - Fires event on change

Usage:
    python 04_sensor_flow.py --environment=pypi run
    python 04_sensor_flow.py --environment=pypi argo-workflows create  # deploy to schedule
"""

from metaflow import (
    step,
    FlowSpec,
    Flow,
    Parameter,
    current,
    card,
    schedule,
    pypi_base,
    project,
)


@schedule(cron="0 8 * * *")  # daily at 8am UTC
@project(name="oni_forecast")
@pypi_base(
    python="3.12",
    packages={
        "requests": "2.32.3",
    },
)
class ONIDataSensor(FlowSpec):
    """
    Sensor that detects new data in the ninodata GitHub repo.

    Checks the last commit SHA for nino_ml.csv via GitHub API.
    If it differs from the previous run, publishes a 'new_oni_data' event.
    """

    repo = Parameter(
        "repo",
        default="ahuang11/ninodata",
        help="GitHub repo in 'owner/name' format.",
    )
    file_path = Parameter(
        "file_path",
        default="nino_ml.csv",
        help="Path to the file within the repo.",
    )
    new_data_event_name = Parameter(
        "event_name",
        default="new_oni_data",
        help="Name of the ArgoEvent to publish when new data is detected.",
    )

    @card(type="blank", id="sensor_status")
    @step
    def start(self):
        """Check GitHub for updates to nino_ml.csv."""
        import requests
        from metaflow.cards import Markdown

        # 1. Get previous state from last successful run
        try:
            prev_run = Flow("ONIDataSensor").latest_successful_run
            self.prev_sha = prev_run.data.current_sha
            self.prev_date = prev_run.data.current_date
        except Exception:
            self.prev_sha = None
            self.prev_date = None

        # 2. Query GitHub API for latest commit on this file
        api_url = (
            f"https://api.github.com/repos/{self.repo}"
            f"/commits?path={self.file_path}&per_page=1"
        )
        resp = requests.get(api_url, timeout=30)
        resp.raise_for_status()
        commits = resp.json()

        if commits:
            self.current_sha = commits[0]["sha"]
            self.current_date = commits[0]["commit"]["committer"]["date"]
            self.current_message = commits[0]["commit"]["message"]
        else:
            self.current_sha = None
            self.current_date = None
            self.current_message = "No commits found"

        # 3. Compare and act
        self.data_changed = (self.prev_sha != self.current_sha)

        run = Flow(current.flow_name)[current.run_id]
        if self.data_changed:
            print(f"NEW DATA DETECTED!")
            print(f"  Previous SHA: {self.prev_sha}")
            print(f"  Current SHA:  {self.current_sha}")
            print(f"  Commit date:  {self.current_date}")
            print(f"  Message:      {self.current_message}")

            # Publish event to trigger downstream flows
            from metaflow.integrations import ArgoEvent
            ArgoEvent(name=self.new_data_event_name).publish()
            run.add_tag("new_data_detected")
        else:
            print(f"No changes detected.")
            print(f"  Current SHA: {self.current_sha}")
            print(f"  Last updated: {self.current_date}")
            run.add_tag("no_change")

        # Card
        status = "🟢 New data detected!" if self.data_changed else "⚪ No changes"
        current.card["sensor_status"].append(Markdown(f"# ONI Data Sensor"))
        current.card["sensor_status"].append(Markdown(f"**Status:** {status}"))
        current.card["sensor_status"].append(Markdown(
            f"- **Repo:** `{self.repo}`\n"
            f"- **File:** `{self.file_path}`\n"
            f"- **Current SHA:** `{self.current_sha}`\n"
            f"- **Previous SHA:** `{self.prev_sha}`\n"
            f"- **Last commit:** {self.current_date}\n"
            f"- **Message:** {self.current_message}\n"
        ))

        self.next(self.end)

    @step
    def end(self):
        """Done."""
        if self.data_changed:
            print(f"Event '{self.new_data_event_name}' published. "
                  f"Downstream flows will be triggered.")
        else:
            print("Nothing to do.")


if __name__ == "__main__":
    ONIDataSensor()
