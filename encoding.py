import pandas as pd
import pm4py
from pm4py.objects.conversion.log import converter as log_converter

# Function used to import a xes file, given its path
def import_xes(file_path):
    log = pm4py.read_xes(file_path)
    log = log_converter.apply(log, variant=log_converter.Variants.TO_EVENT_LOG, parameters={})
    return log

# Function used to create a label encoder
def get_label_encoder(log):
    activity_names = []
    for case_index, case in enumerate(log):
        for event_index, event in enumerate(case):
            activity_names.append(event['concept:name'])
    return {name: idx + 1 for idx, name in
            enumerate(sorted(set(activity_names), key=lambda x: activity_names.index(x)))}

# Function used to print the entire log
def print_all_log(log):
    for case_id, case in enumerate(log):
        print("Case: ", case_id, " ", case.attributes["concept:name"], case[0]["time:timestamp"], "-",
              case[-1]["time:timestamp"], "=" * 50)
        for event_id, event in enumerate(case):
            print(event_id, " ", event["concept:name"])

# Function used to encode a trace using simple index encoding
def encode_case_simple_index(case, prefix_length, label_encoder):
    encoded_case = []
    for event_id, event in enumerate(case):
        encoded_event_name = label_encoder[event["concept:name"]]
        encoded_case.append(encoded_event_name)

    trace = (encoded_case[:prefix_length] + [0] * prefix_length)[:prefix_length]
    trace.append(case.attributes["label"])

    return trace

def decode_case_simple_index(encoded_case, prefix_length, label_encoder):
    inverse_label_encoder = {v: k for k, v in label_encoder.items()}
    decoded_case = []
    for encoded_event in encoded_case:
        if(encoded_event != 0 and encoded_event in inverse_label_encoder):
            encoded_event_name = inverse_label_encoder[encoded_event]
            decoded_case.append(encoded_event_name)

    trace = decoded_case[:prefix_length]

    return trace

# Function used to compute the number of cases in the event log that were executed concurrently with the given trace
def count_concurrent_cases(case, log):
    start_timestamp = case[0]["time:timestamp"]
    end_timestamp = case[-1]["time:timestamp"]

    intersecting_traces = pm4py.filter_time_range(
        log,
        start_timestamp,
        end_timestamp,
        mode='traces_intersecting',
        case_id_key='concept:name',
        timestamp_key='time:timestamp'
    )

    return len(intersecting_traces)

# Function used to compute the average duration of the cases in the event log that were exectued concurrently with the given trace
def count_avg_duration(case, log):
    start_timestamp = case[0]["time:timestamp"]
    end_timestamp = case[-1]["time:timestamp"]

    intersecting_traces = pm4py.filter_time_range(
        log,
        start_timestamp,
        end_timestamp,
        mode='traces_intersecting',
        case_id_key='concept:name',
        timestamp_key='time:timestamp'
    )

    # Compute avg duration
    cumulative_time = 0
    for case_id, case in enumerate(intersecting_traces):
        time = (pd.Timedelta(case[-1]["time:timestamp"] - case[0]["time:timestamp"])).total_seconds()
        cumulative_time += time
        # If the case has only one event, the time will be 0
    return round(cumulative_time / len(intersecting_traces))

# Function used to compute the average amount of unique resources involved in the cases that were executed concurrently with the given trace
def count_avg_resources_concurrent_cases(case, log):
    start_timestamp = case[0]["time:timestamp"]
    end_timestamp = case[-1]["time:timestamp"]

    intersecting_traces = pm4py.filter_time_range(
        log,
        start_timestamp,
        end_timestamp,
        mode='traces_intersecting',
        case_id_key='concept:name',
        timestamp_key='time:timestamp'
    )

    cumulative_resources = 0
    for case_id, case in enumerate(intersecting_traces):
        resources = set()
        for event_id, event in enumerate(case):
            resources.add(event["Resource"])
        cumulative_resources += len(resources)
    return round(cumulative_resources / len(log), 2)

# Function used to compute the average overlapping duration between the given trace and all other traces that are exectured concurrently
def count_avg_overlapping_duration_concurrent_cases(case, log):
    start_timestamp = case[0]["time:timestamp"]
    end_timestamp = case[-1]["time:timestamp"]

    intersecting_traces = pm4py.filter_time_range(
        log,
        start_timestamp,
        end_timestamp,
        mode='traces_intersecting',
        case_id_key='concept:name',
        timestamp_key='time:timestamp'
    )

    overlapping_durations = []

    for case in intersecting_traces:
        case_start = case[0]["time:timestamp"]
        case_end = case[-1]["time:timestamp"]

        overlap_start = max(start_timestamp, case_start)
        overlap_end = min(end_timestamp, case_end)

        if (overlap_start < overlap_end):
            duration = pd.Timedelta(overlap_end - overlap_start).total_seconds()
            overlapping_durations.append(duration)

    if (overlapping_durations):
        retval = round((sum(overlapping_durations) / len(overlapping_durations)), 2)
    else:
        retval = 0

    return retval

# Function used to compute the average number of concurrent cases for each event within the given trace
def count_avg_concurrent_cases_per_event(case, log):
    concurrent_counts = []

    for event in case:
        event_start_timestamp = event["time:timestamp"]
        event_end_timestamp = event_start_timestamp + pd.to_timedelta(event["activity_duration"], unit='s')

        intersecting_traces = pm4py.filter_time_range(
            log,
            event_start_timestamp,
            event_end_timestamp,
            mode='traces_intersecting',
            case_id_key='concept:name',
            timestamp_key='time:timestamp'
        )

        concurrent_counts.append(len(intersecting_traces))

    if (concurrent_counts):
        retval = round((sum(concurrent_counts) / len(concurrent_counts)), 2)
    else:
        retval = 0

    return retval

# Function used to encode the given log using Simple index encoding, with optional intercase features
def simple_index_encode(log, prefix_length, label_encoder, conc_cases=False, avg_dur=False, my_int1=False,
                        my_int2=False, my_int3=False):
    encode_result = []
    for case_id, case in enumerate(log):
        base_encode = encode_case_simple_index(case, prefix_length, label_encoder)

        if (conc_cases):
            # Compute concurrent cases for all the traces
            n = count_concurrent_cases(case, log)
            base_encode.insert(-1, n)
        if (avg_dur):
            # Compute avg duration of concurrent traces
            n = round(count_avg_duration(case, log) / (3600 * 24), 2)
            base_encode.insert(-1, n)
        if (my_int1):
            # Compute avg number of resources used by concurrent traces
            n = count_avg_resources_concurrent_cases(case, log)
            base_encode.insert(-1, n)
        if (my_int2):
            # Compute avg overlapping time duration of concurrent traces
            n = round(count_avg_overlapping_duration_concurrent_cases(case, log) / (3600 * 24), 2)
            base_encode.insert(-1, n)
        if (my_int3):
            # Compute avg number of overlapping traces for each event in a case
            n = count_avg_concurrent_cases_per_event(case, log)
            base_encode.insert(-1, n)

        encode_result.append(base_encode)

    return pd.DataFrame(data=encode_result)


if __name__ == '__main__':
    log = import_xes("./Production_avg_dur_training_0-80.xes")
    label_encoder = get_label_encoder(log)

    # print(encode_case_simple_index(log[0], 5, label_encoder))
    # a = encode_case_simple_index(log[0], 5, label_encoder)
    # print(decode_case_simple_index(a, 5, label_encoder))

    # print(count_concurrent_cases(log[0], log))
    # print(count_avg_duration(log[0], log), " sec")
    # print(round(count_avg_duration(log[0], log)/(3600*24), 2), " days")
    # print(count_avg_resources_concurrent_cases(log[2], log))
    # print(round(count_avg_overlapping_duration_concurrent_cases(log[2], log)/(3600*24), 2), " days")
    # print(count_concurrent_cases_per_event(log[0], log))

    # training_set = simple_index_encode(log, 5, label_encoder, conc_cases=True, avg_dur=True, my_int=True)
    # training_set = simple_index_encode(log, 5, label_encoder, conc_cases=False, avg_dur=True, my_int3=True)
    # print(training_set)
