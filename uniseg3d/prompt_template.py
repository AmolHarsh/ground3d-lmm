
COT_TOKENS = ["<think>", "</think>", "<answer>", "</answer>"]
POINT_TOKEN = ["<|point|>"]
SEG_TOKEN = ["<p>", "</p>", "<SEG>", "<name>", "<sentence>"]

# SEG_INSTRUCTION= '''
# You are an AI visual assistant analyzing a 3D point-cloud scene.
# *Task*: The target classes are {classes}. The given 3D scene is {point}. Identify which classes from the target list are present in this 3D scene and how many instances of each appear.
# *Instructions*:
# - In your thinking process, produce a step-by-step environment description: start with a global overview of the scene, then describe all visible objects relevant to the target classes.
# - Use only information from the scene; do not incorporate external knowledge.
# - In the final answer, list ONLY the classes that are present, using the exact class names from the target list.

# Strict output format (match exactly):
# <think>In this scene, [global environment description]. Specifically, [your step-by-step reasoning describing all relevant objects]</think><answer>There are [count class1, count class2, …]</answer>
# '''


SEG_INSTRUCTION = '''
You are an AI visual assistant analyzing a 3D point-cloud scene.

*Task*
Given the target classes [{classes}] and the 3D scene {point}, determine which target classes are present and how many instances of each appear.

*Instructions*
- In your thinking process, produce a step-by-step environment description: start with a global overview of the scene, then describe every visible object relevant to the target classes.
- The thinking process must start with <think> and end with </think>. The final answer must start with <answer> and end with </answer>.
- Use only information observable in the scene; do not incorporate external knowledge or assumptions.
- In the final answer, list ONLY the classes that are present, using the exact class names from the target list.
- Report counts as integers. Omit any class with a count of 0.
- English ONLY.

Strict output format (match exactly): <think>In this scene, [global environment description]. Specifically, [your step-by-step reasoning describing all relevant objects]</think><answer>There are [count class1, count class2, …]</answer>'''


PART_GLAMM_INSTRUCTION = '''
You are an AI visual assistant analyzing a 3D point-cloud scene. Your task is to provide spatial analysis based on the point-cloud data.
Please include in your response the objects or object parts annotated with <p>object name</p><SEG> or <p>part name</p><SEG>.
When you see <name> trigger, just answer the name of the object or part.
When you see <sentence> trigger, just return the whole sentence and include them into segment tokens like <p>sentence</p><SEG>.
Please answer the question using only the provided point-cloud information: {point}. Don't change the order of the objects or parts, follow the order in the question.
The question is: {question}.
'''


PART_GLAMM_INSTRUCTION_IMG = '''
You are an AI visual assistant analyzing a 3D point-cloud scene. Your task is to provide spatial analysis based on the point-cloud data and the given image.
Please include in your response the objects or object parts annotated with <p>object name</p><SEG> or <p>part name</p><SEG>.
When you see <name> trigger, just answer the name of the object or part.
The point-cloud data is {point}. Don't change the order of the objects or parts, follow the order in the question.
The question is: {question}.
'''

