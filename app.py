import torch
import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

MODEL_ID = 'microsoft/Phi-3.5-mini-instruct'
ADAPTER_ID = 'Chasston/aerollm-phi3.5-mini-asrs'

print('Chargement du modele...')
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
base = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map='auto',
    trust_remote_code=True
)
model = PeftModel.from_pretrained(base, ADAPTER_ID)
model.eval()
print('Modele pret ✅')


def analyze(aircraft, flight_phase, report):
    if not report.strip():
        return "⚠️ Veuillez entrer un rapport d'incident."

    prompt = (
        '<|system|>\n'
        'Tu es un expert en securite aeronautique certifie FAA/EASA. '
        'Analyse ce rapport incident ASRS et fournis : '
        '1) Causes principales 2) Facteurs contributifs 3) Recommandations de securite.<|end|>\n'
        '<|user|>\n'
        f'Aeronef : {aircraft}\n'
        f'Phase de vol : {flight_phase}\n'
        f'Rapport : {report}'
        '<|end|>\n'
        '<|assistant|>\n'
    )

    inputs = tokenizer(prompt, return_tensors='pt')
    input_ids = inputs['input_ids'].to(next(base.parameters()).device)
    attention_mask = inputs['attention_mask'].to(next(base.parameters()).device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=400,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(
        outputs[0][input_ids.shape[1]:],
        skip_special_tokens=True
    )
    return response


examples = [
    [
        'B737-800',
        'Final Approach',
        'GPWS terrain warning activated at 500ft AGL. Crew executed go-around. Investigation revealed incorrect QNH setting during pre-flight.'
    ],
    [
        'A320',
        'Cruise',
        'Unexpected turbulence encountered at FL350. No PIREP available for the area. Two passengers sustained minor injuries due to unfastened seatbelts.'
    ],
    [
        'B777',
        'Takeoff',
        'Bird strike on left engine during rotation. Engine parameters remained normal. Crew declared emergency and returned to departure airport.'
    ]
]

with gr.Blocks(title='AeroLLM', theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🛩️ AeroLLM — Analyse d'incidents aéronautiques ASRS
    **LLM fine-tuné sur 2000 rapports NASA ASRS** · Phi-3.5-mini + QLoRA · [HuggingFace](https://huggingface.co/Chasston/aerollm-phi3.5-mini-asrs) · [GitHub](https://github.com/NaimMG/AeroLLM)
    """)

    with gr.Row():
        with gr.Column(scale=1):
            aircraft = gr.Textbox(
                label='✈️ Aéronef',
                placeholder='Ex: B737-800, A320, B777...',
                value='B737-800'
            )
            flight_phase = gr.Dropdown(
                label='🕐 Phase de vol',
                choices=[
                    'Preflight', 'Taxi', 'Takeoff', 'Climb',
                    'Cruise', 'Descent', 'Approach', 'Final Approach',
                    'Landing', 'Go-Around', 'Parking'
                ],
                value='Final Approach'
            )
            report = gr.Textbox(
                label='📋 Rapport d\'incident (anglais)',
                placeholder='Décrivez l\'incident en anglais...',
                lines=6
            )
            btn = gr.Button('🔍 Analyser', variant='primary')

        with gr.Column(scale=1):
            output = gr.Textbox(
                label='🧠 Analyse AeroLLM',
                lines=15,
                show_copy_button=True
            )

    gr.Examples(
        examples=examples,
        inputs=[aircraft, flight_phase, report],
        label='📂 Exemples d\'incidents'
    )

    btn.click(fn=analyze, inputs=[aircraft, flight_phase, report], outputs=output)

if __name__ == '__main__':
    demo.launch(share=True)