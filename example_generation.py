import numpy as np
import spatialscaper as ss
import os

# Constants
FOREGROUND_DIR = "datasets/sound_event_datasets/FSD50K_FMA"  # Directory with FSD50K foreground sound files
RIR_DIR = (
    "datasets/rir_datasets"  # Directory containing Room Impulse Response (RIR) files
)
ROOM = "metu"  # Initial room setting, change according to available rooms listed below
FORMAT = "mic"  # Output format specifier: could be 'mic' or 'foa'
N_EVENTS_MIN = 1    # identical to audiblelight
N_EVENTS_MAX = 16
DURATION = 60.0  # Duration in seconds of each soundscape
SR = 24000  # SpatialScaper default sampling rate for the audio files
OUTPUT_DIR = "output"  # Directory to store the generated soundscapes

MIN_REF_DB, MAX_REF_DB = -80, -50

# List of possible rooms to use for soundscape generation. Change 'ROOM' variable to one of these:
# "metu", "arni","bomb_shelter", "gym", "pb132", "pc226", "sa203", "sc203", "se203", "tb103", "tc352"
# Each room has a different Room Impulse Response (RIR) file associated with it, affecting the acoustic properties.
TRAIN_ROOMS = ["bomb_shelter", "gym", "pb132", "pc226", "sa203", "sc203"]
TEST_ROOMS = ["se203", "tb103", "tc352"]
SCAPES_PER_TRAIN = 150
SCAPES_PER_TEST = 100

# FSD50K sound classes that will be spatialized include:
# 'femaleSpeech', 'maleSpeech', 'clapping', 'telephone', 'laughter',
# 'domesticSounds', 'footsteps', 'doorCupboard', 'music',
# 'musicInstrument', 'waterTap', 'bell', 'knock'.
# These classes are sourced from the FSD50K dataset, and
# are consistent with the DCASE SELD challenge classes.


# Function to generate a soundscape
def generate_soundscape(index: int, fold_num: int, room_num: int, room: str):
    track_name = f"fold{fold_num}_room{room_num}_mix{index+1:03d}"
    # Initialize Scaper. 'max_event_overlap' controls the maximum number of overlapping sound events.
    ssc = ss.Scaper(
        DURATION,
        FOREGROUND_DIR,
        RIR_DIR,
        FORMAT,
        room,
        background_dir=FOREGROUND_DIR,
        use_room_ambient_noise=False,
        max_event_overlap=2,
        speed_limit=2.0,  # in meters per second
    )

    # static ambient noise
    ssc.ref_db = int(np.random.uniform(MIN_REF_DB, MAX_REF_DB))
    ssc.add_background()

    # Add a random number of foreground events, based on the specified min and max
    n_events = int(np.random.uniform(N_EVENTS_MIN, N_EVENTS_MAX))
    for _ in range(n_events):
        ssc.add_event()  # randomly choosing and spatializing an FSD50K/FMS sound event

    audiofile = os.path.join(OUTPUT_DIR, FORMAT, track_name)
    labelfile = os.path.join(OUTPUT_DIR, "labels", track_name)

    ssc.generate(audiofile, labelfile)


# Main loop for generating soundscapes
def main():
    for train_room_num, train_room in enumerate(TRAIN_ROOMS):
        for n_scape in range(SCAPES_PER_TRAIN):
            print(f"Generating room {train_room_num + 1}/{len(TRAIN_ROOMS)}, soundscape: {n_scape + 1}/{SCAPES_PER_TRAIN}")
            generate_soundscape(n_scape, fold_num=1, room_num=train_room_num, room=train_room)

    for test_room_num, test_room in enumerate(TEST_ROOMS):
        for n_scape in range(SCAPES_PER_TEST):
            print(f"Generating room {test_room_num + 1}/{len(TEST_ROOMS)}, soundscape: {n_scape + 1}/{SCAPES_PER_TEST}")
            generate_soundscape(n_scape, fold_num=2, room_num=test_room_num, room=test_room)


if __name__ == "__main__":
     main()
