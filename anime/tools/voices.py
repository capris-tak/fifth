import json, wave, io
from voicevox_core.blocking import Onnxruntime, OpenJtalk, Synthesizer, VoiceModelFile
import pyopenjtalk, os
DIC = os.path.join(os.path.dirname(pyopenjtalk.__file__), 'open_jtalk_dic_utf_8-1.11')
ort = Onnxruntime.load_once(filename="/opt/vv/voicevox_onnxruntime-linux-x64-1.17.3/lib/libvoicevox_onnxruntime.so.1.17.3")
syn = Synthesizer(ort, OpenJtalk(DIC))
for f in ['0','1','6']:
    with VoiceModelFile.open(f'/opt/vv/vvm/{f}.vvm') as m: syn.load_voice_model(m)
lines = json.load(open('tools/script.json'))
VIS = {'a':'a','i':'i','u':'u','e':'e','o':'o','A':'a','I':'i','U':'u','E':'e','O':'o','N':'n','cl':'n'}
out = []
for L in lines:
    aq = syn.create_audio_query(L['text'], L['style'])
    aq.speed_scale = L.get('speed',1.0); aq.pitch_scale = L.get('pitch',0.0)
    aq.intonation_scale = L.get('intonation',1.0); aq.volume_scale = 1.0
    aq.pre_phoneme_length = 0.05; aq.post_phoneme_length = 0.15
    wav = syn.synthesis(aq, L['style'])
    open(f'audio/voice/{L["id"]}.wav','wb').write(wav)
    with wave.open(io.BytesIO(wav)) as w: dur = w.getnframes()/w.getframerate()
    # mora timeline -> visemes (time in seconds from clip start)
    sp = aq.speed_scale; t = aq.pre_phoneme_length/sp; vis=[]
    for ap in aq.accent_phrases:
        for m in ap.moras:
            c = (m.consonant_length or 0)/sp; v = m.vowel_length/sp
            if c: vis.append([round(t,3), 'n' if m.consonant in ('m','b','p') else 'c']); 
            t += c; vis.append([round(t,3), VIS.get(m.vowel,'n')]); t += v
        if ap.pause_mora:
            vis.append([round(t,3),'x']); t += (ap.pause_mora["vowel_length"] if isinstance(ap.pause_mora,dict) else ap.pause_mora.vowel_length)/sp
    vis.append([round(t,3),'x'])
    out.append({**L,'dur':round(dur,3),'visemes':vis})
    print(L['id'], round(dur,2), 'mora-end', round(t,2))
json.dump(out, open('tools/voices.json','w'), ensure_ascii=False, indent=1)
