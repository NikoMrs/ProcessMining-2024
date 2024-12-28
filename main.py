import pandas as pd
import pm4py
from pm4py.objects.conversion.log import converter as log_converter
from datetime import datetime


def import_xes(file_path):
    log = pm4py.read_xes(file_path)
    log = log_converter.apply(log, variant=log_converter.Variants.TO_EVENT_LOG, parameters={})
    return log

def get_label_encoder(log):
    activity_names=[]
    for case_index,case in enumerate(log):
        for event_index,event in enumerate(case):
            activity_names.append(event['concept:name'])
    return  {name: idx+1 for idx, name in enumerate(sorted(set(activity_names), key=lambda x: activity_names.index(x)))}


def print_all_log(log):
    for case_id, case in enumerate(log):
        print("Case: ", case_id, " ", case.attributes["concept:name"], case[0]["time:timestamp"], "-", case[-1]["time:timestamp"] ,"="*50)
        for event_id, event in enumerate(case):
            print(event_id, " ", event["concept:name"])

# TODO encode_case_simple_index (case, prefix_length, label_encoder) aggiungere label linale con il risultato? protemmo usare -1 per lo Zero padding:
# Compute the Simple Index encoding and cut the encoding at a specific prefix_length,
# if trace_length < prefix_length 0's are added
def encode_case_simple_index (case, prefix_length, label_encoder):
    encoded_case = []
    for event_id, event in enumerate(case):
        encoded_event_name = label_encoder[event["concept:name"]]
        encoded_case.append(encoded_event_name)

    trace = (encoded_case[:prefix_length] + [0] * prefix_length)[:prefix_length]
    trace.append(case.attributes["label"])
    return trace

# TODO decode_case_simple_index (forse)

# TODO count_concurrent_cases (case, log):
def count_concurrent_cases (case, log):
    start_timestamp = case[0]["time:timestamp"]
    end_timestamp = case[-1]["time:timestamp"]
    # print(start_timestamp, " - ", end_timestamp)
    intersecting_traces = pm4py.filter_time_range(
        log,
        start_timestamp,
        end_timestamp,
        mode='traces_intersecting',
        case_id_key='concept:name',
        timestamp_key='time:timestamp'
    )

    return len(intersecting_traces)


# TODO count_avg_duration (case, log):
def count_avg_duration (case, log):
    start_timestamp = case[0]["time:timestamp"]
    end_timestamp = case[-1]["time:timestamp"]
    # print(start_timestamp, " - ", end_timestamp)
    intersecting_traces = pm4py.filter_time_range(
        log,
        start_timestamp,
        end_timestamp,
        mode='traces_intersecting',
        case_id_key='concept:name',
        timestamp_key='time:timestamp'
    )

    # print(intersecting_traces[1])

    # Compute avg duration
    cumulative_time = 0
    for case_id, case in enumerate(intersecting_traces):
        # time = datetime(case[-1]["time:timestamp"]) - datetime(case[0]["time:timestamp"])
        time = (pd.Timedelta(case[-1]["time:timestamp"] - case[0]["time:timestamp"])).total_seconds()
        cumulative_time += time
        # TODO se la traccia ha una sola attivita' allora possiamo guardare la durata della singola attivita', il problema e' che non so in  che unita' di misura e; espressa
        # dur = 0
        # for event_id, event in enumerate(case):
        #     dur += int(event["activity_duration"])
        # print(time, " - ", dur)
    return cumulative_time/len(intersecting_traces)

# TODO count_my_intercase_value: (dobbiamo scegliere una metrica)
# TODO simple_index_encode (log, prefix_length, label_encoder, conc_cases, avg_dur, my_int) come codifico i nuovi valori ottenuti dalle trasformazioni (potremmo aggiungerli al label_encoder):
def simple_index_encode (log, prefix_length, label_encoder, conc_cases, avg_dur, my_int):
    encode_result = []
    for case_id, case in enumerate(log):
        base_encode = encode_case_simple_index(case, 5, label_encoder)

        if (conc_cases):
            # Compute concurrent cases for all the traces
            n = count_concurrent_cases(case, log)
            base_encode.insert(-1, n)
        if (avg_dur):
            # Compute avg duration of concurrent traces
            n = round(count_avg_duration(case, log)/(3600*24), 2)
            base_encode.insert(-1, n)
        if (my_int):
            print("my_int")

        encode_result.append(base_encode)
    return pd.DataFrame(data=encode_result)

if __name__ == '__main__':
    log = import_xes("./Production_avg_dur_training_0-80.xes")
    label_encoder = get_label_encoder(log)
    # print(encode_case_simple_index(log[0], 5, label_encoder))

    # print(count_concurrent_cases(log[0], log))
    # print(round(count_avg_duration(log[0], log)/(3600*24), 2), " days")

    print(simple_index_encode(log, 5, label_encoder, conc_cases=True, avg_dur=True, my_int=False))

    # print(log[0])
    # print_all_log({log[0]})

    # for case_id, case in enumerate(log):
    #     print(encode_case_simple_index(case, 5, label_encoder))
