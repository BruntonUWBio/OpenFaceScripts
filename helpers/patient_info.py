def get_patient_names(vid_dirs):
    patient_map = set()

    for vid_dir in vid_dirs:
        patient = vid_dir.split('_')[0]
        patient_map.add(patient)

    return list(patient_map)


def patient_day_session(vid_dir: str):
    split_name = vid_dir.split('_')
    patient = split_name[0]
    day = split_name[1]
    session = split_name[2]

    return patient, day, session
