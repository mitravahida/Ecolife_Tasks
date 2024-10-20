import json
import sys
import numpy as np
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
# gill sans
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Gill Sans"
import sys
sys.path.append("..")
import fire
import utils

def read_json_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
        return data
    
def verify(
    window_size: int=20,
    interval: int=24*60,
):
    traces, trace_function_names,original_names = utils.read_selected_traces()
    sum_invoke = 0
    for j in range(window_size, window_size+interval):
         for i in range(len(traces)):
                if int(traces[i][j])!=0:
                    sum_invoke+=int(traces[i][j])

    genetic_carbon = read_json_file("./results/genetic_algorithm/carbon.json")
    genetic_st = read_json_file("./results/genetic_algorithm/st.json")

    sum_carbon_ga = 0
    sum_st_ga = 0

    for i in range(len(traces)):
        for _, inner_dict in genetic_carbon[i].items():
            if isinstance(inner_dict, dict) and "carbon" in inner_dict:
                sum_carbon_ga += inner_dict["carbon"]
        for _, inner_dict in genetic_st[i].items():
            if isinstance(inner_dict, dict) and "st" in inner_dict:
                sum_st_ga += inner_dict["st"]

    eco_carbon = read_json_file("./results/eco_life/carbon.json")
    eco_st = read_json_file("./results/eco_life/st.json")

    sum_carbon_eco = 0
    sum_st_eco = 0

    for i in range(len(traces)):
        for _, value in eco_carbon[i].items():
            sum_carbon_eco += value["carbon"]
        for _, value in eco_st[i].items():
            sum_st_eco += value["st"]

    avg_carbon_ga = sum_carbon_ga / sum_invoke
    avg_st_ga = sum_st_ga / sum_invoke
    avg_carbon_eco = sum_carbon_eco / sum_invoke
    avg_st_eco = sum_st_eco / sum_invoke

    algorithms = ['Genetic Algorithm', 'Eco-Life']
    avg_carbon = [avg_carbon_ga, avg_carbon_eco]
    avg_service_time = [avg_st_ga, avg_st_eco]

    x = np.arange(len(algorithms))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width / 2, avg_carbon, width, label='CO$_2$ Footprint (g)', color='skyblue', edgecolor='black')
    rects2 = ax.bar(x + width / 2, avg_service_time, width, label='Service Time (s)', color='lightcoral', edgecolor='black')

    ax.set_xlabel('Algorithms', fontsize=14)
    ax.set_ylabel('Average Values', fontsize=14)
    ax.set_title('Comparison of Genetic Algorithm vs Eco-Life Algorithm', fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(algorithms, fontsize=12)
    ax.legend()

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=12)

    autolabel(rects1)
    autolabel(rects2)

    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig("comparison_genetic_vs_eco.pdf", bbox_inches='tight')

if __name__ == "__main__":
    fire.Fire(verify)
