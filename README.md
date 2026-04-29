The sweep + FVG sequence is a micro state machine: state 0 = waiting for sweep → state 1 = sweep detected → state 2 = FVG found → state 3 = entered. Coding this explicitly reveals why the setup is rare — all three states must resolve in the same 60-minute window
Forward-looking simulation is the hardest part of any backtester — you have to walk bar by bar without "peeking" at future prices, and you have to handle the case where neither target nor stop is reached
Timeouts matter — in a 2:1 RR system, a trade that sits unresolved for 2 hours and then closes slightly negative is essentially a loss. Logging these separately prevents false win rate inflation
The Sharpe ratio at daily granularity with only 1-3 trades per day can be noisy — a higher sample size (more days) gives more reliable estimates
Combining zoneinfo for correct timezone handling with numpy forward-scan loops produces backtesting code that is both fast and correct across DST transitions


Tech stack

yfinance — 5-minute intraday bars
pandas + numpy — swing detection, FVG detection, simulation
zoneinfo — correct NY EST/EDT timezone handling
matplotlib — 6-panel dark dashboard
pytest — unit tests for windows, FVGs, swings, simulation, and metrics


similar to ICT 2022 MARKET MAKER STrategy
