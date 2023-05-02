#### EXAMPLE -- BASELINE ####

baseline-davinci3-test:
	python main.py -t all -f test -p 0 -m gpt_davinci-003 --test

baseline-davinci3:
	python main.py -t all -f test -p 0 -m gpt_davinci-003

evaluation-davinci3-baseline:
	python main.py -t all -f test -p 4 -m gpt_davinci-003 --baseline

#### EXAMPLE -- FARM, CREDIBLE, 3 SNIPPETS ####

foveation-davinci3-test:
	python main.py -t all -f test -p 1 -m gpt_davinci-003 --test

foveation-davinci3:
	python main.py -t all -f test -p 1 -m gpt_davinci-003

attribution-davinci3-credible-test:
	python main.py -t all -f test -p 2 -m gpt_davinci-003 -a google_credible --test

attribution-davinci3-credible:
	python main.py -t all -f test -p 2 -m gpt_davinci-003 -a google_credible

rationalization-davinci3-credible-snippet3-test:
	python main.py -t all -f test -p 3 -m gpt_davinci-003 -a google_credible -s 3 -e 16 --test

rationalization-davinci3-credible-snippet3:
	python main.py -t all -f test -p 3 -m gpt_davinci-003 -a google_credible -s 3 -e 16

evaluation-davinci3-credible-snippet3:
	python main.py -t all -f test -p 4 -m gpt_davinci-003 -a google_credible -s 3