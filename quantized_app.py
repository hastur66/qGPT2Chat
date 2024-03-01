from tensorflow.lite.python import interpreter
import tensorflow_text as tf_text
import numpy as np

import gradio as gr
from gradio.themes.utils.colors import Color


DESCRIPTION = """
# gpt2-interface Chat 🗨️
"""


with open("quant_generate.tflite", "rb") as f:
    quant_generate_tflite = f.read()
   

def run_inference(input, generate_tflite):
  """
    Run inference with TFLite.
    """
  interp = interpreter.InterpreterWithCustomOps(
      model_content=generate_tflite,
      custom_op_registerers=tf_text.tflite_registrar.SELECT_TFTEXT_OPS
  )

  interp.get_signature_list()

  generator = interp.get_signature_runner("serving_default")
  output = generator(prompt=np.array([input]))
  return output


def chat(user_input, history, model=quant_generate_tflite):
    """
    chat function.
    """
    output = run_inference(user_input, model)
    return output


text_color = "#FFFFFF"
app_background = "#0A0A0A"
user_inputs_background = "#193C4C"
widget_bg = "#000100"
button_bg = "#141414"

dark = Color(
    name="dark",
    c50="#F4F3EE",  # not sure
    # all text color:
    c100=text_color, # Title color, input text color, and all chat text color.
    c200=text_color, # Widget name colors (system prompt and "chatbot")
    c300="#F4F3EE", # not sure
    c400="#F4F3EE", # Possibly gradio link color. Maybe other unlicked link colors.
    # suggestion text color...
    c500=text_color, # text suggestion text. Maybe other stuff.
    c600=button_bg,#"#444444", # button background color, also outline of user msg.
    # user msg/inputs color:
    c700=user_inputs_background, # text input background AND user message color. And bot reply outline.
    # widget bg.
    c800=widget_bg, # widget background (like, block background. Not whole bg), and bot-reply background.
    c900=app_background, # app/jpage background. (v light blue)
    c950="#F4F3EE", # not sure atm. 
)

with gr.Blocks(theme=gr.themes.Monochrome(
               font=[gr.themes.GoogleFont("Montserrat"), "Arial", "sans-serif"],
               primary_hue="sky",  # when loading
               secondary_hue="sky", # something with links
               neutral_hue="dark"),) as demo:

    gr.Markdown(DESCRIPTION)
    chatbot = gr.ChatInterface(fn=chat)

demo.queue(api_open=False).launch(show_api=True, share=True)
