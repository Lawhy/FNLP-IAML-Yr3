import sys, re, importlib, traceback, os
from contextlib import contextmanager

msg=None
if len(sys.argv)==1:
  msg='Solution filename missing'
elif len(sys.argv)>2:
  msg='No arguments allowed, just a file name'

if msg is None:
  modFilename=sys.argv[1]
  mres=re.match("([^.]*)\.py$",modFilename)
  if mres is None:
    msg='%s is not a valid python filename'%modFilename

if msg is not None:
  print("""%s
  Usage (in fnlp environment): python idrive.py [yoursolution].py > [your UUN].txt"""%msg,file=sys.stderr)
  exit(1)

modName=mres.group(1)

@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:  
            yield
        finally:
            sys.stdout = old_stdout

sys.path=['.']+sys.path
with suppress_stdout():
  try:
    mod=importlib.import_module(modName)
  except (ModuleNotFoundError, ImportError) as e:
    print("Filename %s must be importable: %s"%(modFilename,e),file=sys.stderr)
    print("Search path was: %s"%sys.path,file=sys.stderr)
    exit(2)
  print("Starting run, please be patient, close any graphs which pop up!",file=sys.stderr)
  sys.stderr.flush()
  mod.answers()

ans=['answer1a',
     'answer1b',
     ('answer2','repr(answer2)'),
     'answer3a',
     'answer3b',
     ('answer4a1','len(answer4a)'),
     ('answer4a2','answer4a[:100]'),
     ('answer4b1','len(answer4b)'),
     ('answer4b2','answer4b[:100]'),
     'answer5a',
     'answer5b',
     ('answer6','repr(answer6)'),
     ('brown_bigram_model_len','brown_bigram_model._N'),
     ('answer8len','len(answer8)'),
     ('answer8best','answer8[:10]'),
     ('answer8worst','answer8[-10:]'),
     ('answer9','repr(answer9)'),
     ('answer10mean','answer10[0]'),
     ('answer10stdev','answer10[1]'),
     ('answer10_ascii_best','answer10[2][:10]'),
     ('answer10_ascii_worst','answer10[2][-10:]'),
     ('answer10_nonEnglish_best','answer10[3][:10]'),
     ('answer11best','answer11[:10]'),
     ('answer11worst','answer11[-10:]')]
for an in ans:
    if type(an) is str:
        (aname,aval)=(an,an)
    else:
        (aname,aval)=an
    try:
      if type(aval) is tuple:
        av=aval[0]
      else:
        av=eval(aval,mod.__dict__)
    except Exception as e:
      print('%s="""%s threw Exception: %s"""'%(aname,aval,traceback.format_exc().replace('"',"'")),file=sys.stderr)
      continue
    print("%s=%s"%(aname,av))
