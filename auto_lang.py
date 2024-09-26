import subprocess

languages = ['heb']

for language in languages:
    print(f"Running for language: {language}")
    subprocess.run(
                    ["python", "auto1.py", "--language", language], 
                    check=True
                )

