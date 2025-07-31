rm -rf /Users/kormedi/Documents/WorkPlace/github/minuee-llm-test/*
rsync -rub --delete --exclude={'node_modules','logs','.git','.venv'} /Users/kormedi/Documents/WorkPlace/bitbucket/llm_test/* /Users/kormedi/Documents/WorkPlace/github/minuee-llm-test/
