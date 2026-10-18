"""Step 1 of the pipeline: Apple Health export.xml -> small watch-only CSVs.

Streams the XML (it is ~270 MB) and keeps only records whose source is an Apple Watch.
Output goes to data_private/watch/ which is git-ignored (raw health data stays local).

usage: python src/parse_health.py ../apple_health_export/export.xml data_private/watch
"""
import csv
import os
import sys
import xml.etree.ElementTree as ET

KEEP = {
    "HKQuantityTypeIdentifierHeartRate": "heart_rate",
    "HKQuantityTypeIdentifierHeartRateVariabilitySDNN": "hrv",
    "HKQuantityTypeIdentifierRestingHeartRate": "resting_hr",
    "HKQuantityTypeIdentifierStepCount": "steps",
    "HKQuantityTypeIdentifierActiveEnergyBurned": "active_energy",
    "HKQuantityTypeIdentifierRespiratoryRate": "resp_rate",
    "HKQuantityTypeIdentifierAppleSleepingWristTemperature": "wrist_temp",
    "HKQuantityTypeIdentifierOxygenSaturation": "spo2",
    "HKCategoryTypeIdentifierSleepAnalysis": "sleep",
    "HKQuantityTypeIdentifierPhysicalEffort": "physical_effort",       # watch-only, back to Sep 14
    "HKQuantityTypeIdentifierAppleExerciseTime": "exercise_min",
    "HKCategoryTypeIdentifierAppleStandHour": "stand_hour",
    "HKQuantityTypeIdentifierTimeInDaylight": "daylight_min",
    "HKQuantityTypeIdentifierEnvironmentalAudioExposure": "noise_db",
    "HKQuantityTypeIdentifierWalkingHeartRateAverage": "walking_hr",
    "HKQuantityTypeIdentifierVO2Max": "vo2max",
    "HKCategoryTypeIdentifierMindfulSession": "mindful",
}
FIELDS = ["startDate", "endDate", "value", "unit", "sourceName"]
WORKOUT_FIELDS = ["workoutActivityType", "startDate", "endDate", "duration", "durationUnit", "sourceName"]


def is_watch(source):
    return source is not None and "watch" in source.lower()


def parse(xml_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    writers, files, counts = {}, [], {}

    def writer(name, header):
        if name not in writers:
            f = open(os.path.join(out_dir, f"{name}.csv"), "w", newline="")
            files.append(f)
            writers[name] = csv.writer(f)
            writers[name].writerow(header)
            counts[name] = 0
        counts[name] += 1
        return writers[name]

    for _, el in ET.iterparse(xml_path):
        if el.tag == "Record" and el.get("type") in KEEP and is_watch(el.get("sourceName")):
            writer(KEEP[el.get("type")], FIELDS).writerow([el.get(k) for k in FIELDS])
        elif el.tag == "Workout" and is_watch(el.get("sourceName")):
            writer("workouts", WORKOUT_FIELDS).writerow([el.get(k) for k in WORKOUT_FIELDS])
        if el.tag in ("Record", "Workout"):
            el.clear()  # free memory
    for f in files:
        f.close()
    return counts


if __name__ == "__main__":
    print(parse(sys.argv[1], sys.argv[2]))
