PY ?= python3
S = scripts

.PHONY: all install test quick table1 fig1 fig2 fig3 checks clean

all: table1 fig1 fig2 fig3 checks        ## reproduce every result of the paper (~3 min)

install:                                  ## editable install with test dependencies
	$(PY) -m pip install -e ".[test]"

test:                                     ## numerical checks of the identities (Appendix A)
	$(PY) -m pytest -q

quick:                                    ## smoke run of every script (~1 min)
	for f in $(S)/0*.py; do $(PY) $$f --quick || exit 1; done

table1:
	$(PY) $(S)/01_table1_budget_drift.py
fig1:
	$(PY) $(S)/02_fig1_mobility.py
fig2:
	$(PY) $(S)/05_fig2_reciprocity_test.py
fig3:
	$(PY) $(S)/06_fig3_loop_count_mc.py
	$(PY) $(S)/07_fig3_threshold.py
	$(PY) $(S)/08_plot_fig3.py
checks:
	$(PY) $(S)/03_power_balance.py
	$(PY) $(S)/04_sign_cross_mobility.py

clean:
	rm -rf results/*.json results/*.tex figures/*.pdf figures/*.png
