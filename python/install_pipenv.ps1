pip install --upgrade pip
pip install pipenv
$pythonUserPath = python -m site --user-site
$pythonUserPath = $pythonUserPath.Replace('site-packages', 'Scripts')
$ENV:Path += ";" + $pythonUserPath
$userPath = [System.Environment]::GetEnvironmentVariable("Path", "User")
$userPath += ";" + $pythonUserPath
[System.Environment]::SetEnvironmentVariable("Path", $userPath, "User")
$ENV:PIPENV_VENV_IN_PROJECT = '.venv'
[System.Environment]::SetEnvironmentVariable("PIPENV_VENV_IN_PROJECT", ".venv", "User")
