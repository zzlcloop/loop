@echo off
setlocal enabledelayedexpansion

set PACKAGES=^
pickleshare==0.7.4 ^
Pillow==10.3.0 ^
prompt-toolkit==1.0.15 ^
protobuf==3.18.3 ^
pyasn1==0.4.8 ^
pyasn1-modules==0.2.8 ^
Pygments==2.15.0 ^
pyparsing==2.4.7 ^
python-dateutil==2.6.1 ^
PyYAML==5.4.1 ^
pyzmq==16.0.2 ^
requests==2.32.0 ^
requests-oauthlib==1.3.0 ^
rsa==4.7.2 ^
scikit-learn==1.5.0 ^
scipy==1.10.0 ^
simplegeneric==0.8.1 ^
six==1.16.0 ^
smart-open==5.1.0 ^
tensorboard==2.0.2 ^
tensorflow==2.12.1 ^
tensorflow-estimator==2.0.1 ^
termcolor==1.1.0 ^
threadpoolctl==2.1.0 ^
tornado==6.4.2 ^
traitlets==4.3.2 ^
typing-extensions==3.10.0.0 ^
urllib3==1.26.19 ^
wcwidth==0.1.7 ^
Werkzeug==3.0.6 ^
wincertstore==0.2 ^
wrapt==1.12.1 ^
zipp==3.19.1

for %%p in (%PACKAGES%) do (
    pip install %%p -i https://pypi.tuna.tsinghua.edu.cn/simple
)

endlocal
