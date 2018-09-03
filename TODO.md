get_yield in a portfolio (SPY|BND) isn't calculated correctly, since we don't take into account the proper number of units each ticket should
have in order to be equal weighted. To caclualted properly, a "portfolio" needs to be an actual portfolio, with number of units etc.

To also calcuated capital gains tax for portfolio flow, we need each portfolio to also track purchase lots.

