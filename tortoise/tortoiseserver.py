import argparse
import os

import torch
import torchaudio

import time
import sys

#sys.path.append("C:/Users/Jarro/Desktop/fast tortoise/tortoise-tts-fastest")

from api import TextToSpeech, MODELS_DIR
from utils.audio import load_voices
#from models.vocoder import VocConf



parser = argparse.ArgumentParser()
parser.add_argument('--text', type=str, help='Text to speak.', default="The expressiveness of autoregressive transformers is literally nuts! I absolutely adore them.")
parser.add_argument('--voice', type=str, help='Selects the voice to use for generation. See options in voices/ directory (and add your own!) '
                                                'Use the & character to join two voices together. Use a comma to perform inference on multiple voices.', default='random')
parser.add_argument('--preset', type=str, help='Which voice preset to use.', default='standard')
parser.add_argument('--use_deepspeed', type=bool, help='Use deepspeed for speed bump.', default=True)
parser.add_argument('--output_path', type=str, help='Where to store outputs.', default='results/')
parser.add_argument('--model_dir', type=str, help='Where to find pretrained model checkpoints. Tortoise automatically downloads these to .models, so this'
                                                    'should only be specified if you have custom checkpoints.', default=MODELS_DIR)
parser.add_argument('--candidates', type=int, help='How many output candidates to produce per-voice.', default=1)
parser.add_argument('--seed', type=int, help='Random seed which can be used to reproduce results.', default=None)
parser.add_argument('--produce_debug_state', type=bool, help='Whether or not to produce debug_state.pth, which can aid in reproducing problems. Defaults to true.', default=True)
parser.add_argument('--cvvp_amount', type=float, help='How much the CVVP model should influence the output.'
                                                        'Increasing this can in some cases reduce the likelihood of multiple speakers. Defaults to 0 (disabled)', default=.0)
parser.add_argument('--temperature', type=float, help='The softmax temperature of the autoregressive model.', default=.8)

parser.add_argument('--autoregressive_samples', type=int, help='umber of samples taken from the autoregressive model, all of which are filtered using CLVP. As Tortoise is a probabilistic model, more samples means a higher probability of creating something "great".')
parser.add_argument('--diffusion_iterations', type=int, help='Number of diffusion steps to perform. [0,4000]. More steps means the network has more chances to iteratively refine the output, which should theoretically mean a higher quality output. Generally a value above 250 is not noticeably better, however.')

args = parser.parse_args()





#if __name__ == '__main__':


if (hasattr(args, "autoregressive_samples") and args.autoregressive_samples is not None) or (hasattr(args, "diffusion_iterations") and args.diffusion_iterations is not None):
    del args.preset
if hasattr(args, "preset"):
    del args.autoregressive_samples
    del args.diffusion_iterations


os.makedirs(args.output_path, exist_ok=True)
#print(f'use_deepspeed do_tts_debug {use_deepspeed}')
#tts2 = TextToSpeech(models_dir=args.model_dir, autoregressive_batch_size=1, high_vram=True, enable_redaction=False)
tts = TextToSpeech(models_dir=args.model_dir, autoregressive_batch_size=1, enable_redaction=False, half=False, kv_cache=True, use_deepspeed=True)

#vocoder=VocConf.BigVGAN,
#selected_voices = args.voice.split(',')

voices = {}
def get_voice(voice):
    global voices
    if(voice not in voices):
        if not os.path.isfile(f"{voice}.pth"):
            selected_voice = voice

            voice_sel = [selected_voice]
            voice_samples, conditioning_latents = load_voices(voice_sel)

            latents = tts.get_conditioning_latents(
                voice_samples,
                return_mels=False,
            )
            torch.save(latents, f"{voice}.pth")
        else: 
            latents = torch.load(f"{voice}.pth")[:2]
        voices[voice] = latents
    return voices[voice]



def infer(text, voice):


    latents = get_voice(voice)

    #input("done 1")
    #for i in range(0, 3):
    t = time.time()

    #t = time.time()
    torch.cuda.empty_cache()
    print(f"took {time.time()-t} to clear cache")

    # gen, dbg_state = tts.tts(args.text, k=args.candidates, conditioning_latents=latents,
    #     use_deterministic_seed=args.seed, return_deterministic_state=True, cvvp_amount=args.cvvp_amount,
    #     temperature=args.temperature, num_autoregressive_samples=args.autoregressive_samples, diffusion_iterations=args.diffusion_iterations)
    #use_deterministic_seed=6,
    gen = tts.tts(text, k=args.candidates, conditioning_latents=latents, return_deterministic_state=False, cvvp_amount=args.cvvp_amount,
        temperature=args.temperature, num_autoregressive_samples=args.autoregressive_samples, diffusion_iterations=args.diffusion_iterations, voice=1 if voice == "musk" else 0)

    print(f"took {time.time()-t}")

    t = time.time()
    timestamp = int(time.time())
    outdir = f"{args.output_path}/{voice}/{timestamp}/"

    os.makedirs(outdir, exist_ok=True)

    with open(os.path.join(outdir, f'input.txt'), 'w') as f:
        f.write(args.text)

    # if isinstance(gen, list):
    #     for j, g in enumerate(gen):
    #         torchaudio.save(os.path.join(outdir, f'{int(t)}_{j}.wav'), g.squeeze(0).cpu(), 24000)
    # else:
    #    #print(gen.squeeze(0).cpu().shape)
    outfile = os.path.join(outdir, f'{int(t)}.wav')
    torchaudio.save(outfile, gen[0].squeeze(0).unsqueeze(0).cpu(), 24000)

    print(f"took {time.time()-t} to save")

    return outfile
    # if args.produce_debug_state:
    #     os.makedirs('debug_states', exist_ok=True)
    #     torch.save(dbg_state, f'debug_states/do_tts_debug_{selected_voice}.pth')




#get_voice("walter3")
#get_voice("mike10")
#get_voice("zoil4")





from http.server import BaseHTTPRequestHandler, HTTPServer, ThreadingHTTPServer
import time
from urllib.parse import urlparse
from urllib.parse import parse_qs
import http.server
import threading
import shutil
import os

hostName = ""
serverPort = 1982



class MyServer(BaseHTTPRequestHandler):
    def do_GET(self):
        print('request')
        query_components = parse_qs(urlparse(self.path).query)
        """if "message" in query_components:
            print("yes")
            print(query_components["message"])
            os.system("rm Hello.wav")
            os.system("generateTTS.ahk \""+query_components["message"]+"\"")"""

        print("-"*50)
        line = query_components["message"][0]
        voice = query_components["voice"][0]

        self.send_response(200)
        self.send_header("Keep-Alive", True)
        #if line == "":
        #    continue
        #filename = end_to_end_infer(line, pronounciation_dictionary, show_graphs)
        filename = infer(line, voice)
        #filename = "forsen.wav"
        #print()
        #name = query_components["name"][0]


        # """self.send_response(200)
        # self.send_header("Content-type", "text/html")
        # self.end_headers()
        # self.wfile.write(bytes("test.wav", "utf-8"))"""
        print(self.headers)
        with open(filename, 'rb') as f:
            self.send_header("Content-Type", 'application/octet-stream')
            self.send_header("Content-Disposition", 'attachment; filename="{}"'.format(os.path.basename(filename)))
            fs = os.fstat(f.fileno())
            self.send_header("Content-Length", str(fs.st_size))
            self.end_headers()
            shutil.copyfileobj(f, self.wfile)
        """self.wfile.write(bytes("<html><head><title>https://pythonbasics.org</title></head>", "utf-8"))
        self.wfile.write(bytes("<p>Request: %s</p>" % self.path, "utf-8"))
        self.wfile.write(bytes("<body>", "utf-8"))
        self.wfile.write(bytes("<p>This is an example web server.</p>", "utf-8"))
        self.wfile.write(bytes("</body></html>", "utf-8"))"""
    def do_POST(self):
        # Doesn't do anything with posted data
        content_length = int(self.headers['Content-Length']) # <--- Gets the size of data
        post_data = self.rfile.read(content_length) # <--- Gets the data itself

if __name__ == "__main__":
    """thread1 = threading.Thread(target=fileserver, args=())
    #thread1.daemon = True
    thread1.start()"""

    webServer = ThreadingHTTPServer((hostName, serverPort), MyServer)
    print("Server started http://%s:%s" % (hostName, serverPort))

    try:
        webServer.serve_forever()
    except KeyboardInterrupt:
        pass
    print("test")
    webServer.server_close()
    print("Server stopped.")

































