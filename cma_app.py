# Design a streamlit app perform the same analysis
import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import altair as alt

# Set the page configuration
st.set_page_config(page_title='Crossover MA Strategy Analysis', layout='wide')

st.title('Crossover MA Strategy Analysis')
st.write('This app performs a crossover moving average strategy analysis \
         comparing it with a buy and hold strategy.')

with st.sidebar:
    st.header('Select ticker and date range')
    with st.form(key='parameters_form'):
        # Select the stock ticker
        stock = st.text_input('Stock Ticker', 'GPS')

        # Select the start and end date
        start = st.date_input('Start Date', pd.to_datetime('2018-01-01'))
        end = st.date_input('End Date', pd.to_datetime(np.datetime64('today')))

        # Define the initial capital
        initial_capital = st.number_input('Initial Capital', 1000)

        # Include an ENTER button
        submit_button = st.form_submit_button(label='Do the analysis!')

if submit_button:
    # Download the stock data
    try:
        stock_data = yf.download(stock,start,end)
    except ValueError:
        st.write('The stock ticker is not valid. Please try again.')

    # Drop all the colymns except for the adjusted close price
    stock_data = stock_data[['Adj Close']]

    # Calculate daily returns
    stock_data['Daily Returns'] = stock_data['Adj Close'].pct_change()

    # Test different short and long windows iteratively
    short_window = range(15, 60, 1)
    long_window = range(60, 300, 5)

    # Initialize the DataFrame `results`
    results = pd.DataFrame(index=short_window, columns=long_window)

    # Create a copy of stock data to work on
    stock_data_iter = stock_data.copy()

    # Loop through the short window
    for short in short_window: 
        # Loop through the long window
        for long in long_window:
            # Initialize the `signals` DataFrame with the `signal` column
            signals = pd.DataFrame(index=stock_data_iter.index)
            signals['signal'] = 0.0

            # Create short simple moving average over the short window
            signals['short_mavg'] = stock_data_iter['Adj Close'].rolling(
                window=short, center=False).mean()

            # Create long simple moving average over the long window
            signals['long_mavg'] = stock_data_iter['Adj Close'].rolling(
                window=long, center=False).mean()

            # Create signals
            signals['signal'][short:] = np.where(signals['short_mavg'][short:] 
                                                > signals['long_mavg'][short:], 1.0, 0.0)

            # Generate trading orders
            signals['positions'] = signals['signal'].diff()

            # Calculate the strategy returns
            stock_data_iter['CMA Strategy Returns'] = signals['signal'].shift(1) * \
                stock_data_iter['Daily Returns']

            # Calculate the cumulative strategy returns
            stock_data_iter['Cumulative CMA Strategy Returns'] = (
                1 + stock_data_iter['CMA Strategy Returns']).cumprod()

            # Save the results in the DataFrame
            results[long][short] = stock_data_iter['Cumulative CMA Strategy Returns'][-1]

    # Find the position of the best strategy
    best_strat = np.unravel_index(np.argmax(results.values, axis=None), results.values.shape)
    best_short = short_window[best_strat[0]]
    best_long = long_window[best_strat[1]]

    # Initialize the `signals` DataFrame with the `signal` column
    signals = pd.DataFrame(index=stock_data.index)
    signals['signal'] = 0.0

    # Create short simple moving average over the short window
    signals['short_mavg'] = stock_data['Adj Close'].rolling(window=best_short, center=False).mean()

    # Create long simple moving average over the long window
    signals['long_mavg'] = stock_data['Adj Close'].rolling(window=best_long, center=False).mean()

    # Create signals
    signals['signal'][best_short:] = np.where(signals['short_mavg'][best_short:] 
                                            > signals['long_mavg'][best_short:], 1.0, 0.0)

    # Generate trading orders
    signals['positions'] = signals['signal'].diff()

    # Analysis header
    st.header('Analysis of ' + stock)

    # Introduce the overall stock prices before showing strategy
    st.write(f'The stock prices between **{str(start)}** and **{str(end)}** are shown below:')
    st.line_chart(stock_data['Adj Close'], color='#1f78b4')

    st.write(f"According to our calculations, the best short window is **{best_short}** days and \
             the best long window is **{best_long}** days. The crossover strategy is shown below \
                for a initial capital of **${initial_capital}**. The :green[**green**] triangles indicate \
                    buy signals, while the :red[**red**] triangles indicate sell signals.")

    # Combine all necessary data into a single DataFrame for convenience
    chart_data = pd.DataFrame({
        'Date': stock_data.index,
        'Adj Close': stock_data['Adj Close'],
        'Short Moving Average': signals['short_mavg'],
        'Long Moving Average': signals['long_mavg'],
        'Position': signals['positions']
    })

    # Base chart for the lines
    base = alt.Chart(chart_data).encode(x='Date:T')

    # Legend for the chart
    legend_1 = base.mark_line().encode(
        y=alt.Y('Adj Close:Q', axis=alt.Axis(title='')),
        color=alt.Color('variable:N',
                        legend = alt.Legend(title=''),
                        scale=alt.Scale(range=['#d9d9d9',
                                               '#1f78b4',
                                               '#e45756'],
                                        domain=['Adj Closing Price',
                                                'Short Moving Average',
                                                'Long Moving Average']))
    )

    # Line chart for closing prices and moving averages
    line_chart = base.mark_line().encode(
        alt.Y('Adj Close:Q', axis=alt.Axis(title='Stock price in $')),
        color=alt.value('#d9d9d9')
    ) + base.mark_line().encode(
        alt.Y('Short Moving Average:Q'),
        color=alt.value('#1f78b4')
    ) + base.mark_line().encode(
        alt.Y('Long Moving Average:Q'),
        color=alt.value('#e45756')
    )
    
    # Points for buy/sell signals
    buy_signals = base.mark_point(shape='triangle-up', size=100, color='green', strokeWidth=6).encode(
        y='Short Moving Average:Q'
    ).transform_filter(
        alt.datum.Position == 1
    )

    sell_signals = base.mark_point(shape='triangle-down', size=100, color='red', strokeWidth=6).encode(
        y='Short Moving Average:Q'
    ).transform_filter(
        alt.datum.Position == -1
    )

    # Combine charts
    final_chart = (legend_1 + line_chart + buy_signals + sell_signals).properties(
        title=f'Long and short siginals for Crossover MA Strategy - {stock}').interactive()

    # Display chart in Streamlit
    st.altair_chart(final_chart, use_container_width=True)

    # Locate the first buy signal and return the next day
    first_buy_signal = signals[signals['positions'] == 1.0].index[0] + pd.DateOffset(1)

    # Calculate Cumulative Strategy Returns for the Buy and Hold Strategy in $
    stock_data.loc[first_buy_signal:, 'Cumulative Buy and Hold Strategy Returns'] = (1 + stock_data.loc[first_buy_signal:, 'Daily Returns']).cumprod()

    # Calculate Strategy Returns for the Crossover Strategy in $
    stock_data.loc[first_buy_signal:, 'CMA Strategy Returns'] = signals['signal'].shift(1) * stock_data['Daily Returns']

    # Calculate Cumulative Strategy Returns
    stock_data.loc[first_buy_signal:, 'Cumulative CMA Strategy Returns'] = (1 + stock_data['CMA Strategy Returns']).cumprod()

    # Create a DataFrame `positions`
    positions = pd.DataFrame(index=stock_data.index).fillna(0.0)

    # Add `holdings` to portfolio
    positions['holding_real'] = stock_data['Cumulative Buy and Hold Strategy Returns'] * initial_capital

    # Daily return
    positions['return_real'] = positions['holding_real'] - positions['holding_real'].shift(1)

    # Add `holdings` to portfolio
    positions['cma_strategy'] = stock_data['Cumulative CMA Strategy Returns'] * initial_capital

    # Daily return
    positions['return_strategy'] = positions['cma_strategy'] - positions['cma_strategy'].shift(1)

    # Prepare the data
    chart_data = pd.DataFrame({
        'Date': positions.index,
        'CMA Strategy': positions['cma_strategy'],
        'Buy and Hold': positions['holding_real'],
        'Position': signals['positions']
    })

    # Base chart for line plots
    base = alt.Chart(chart_data).encode(x='Date:T')

    # Legend for the chart
    legend_2 = base.mark_line().encode(
        y=alt.Y('CMA Strategy:Q', axis=alt.Axis(title='')),
        color=alt.Color('variable:N',
                        legend = alt.Legend(title=''), 
                        scale=alt.Scale(range=['#4c78a8',
                                               'grey'],
                                         domain=['CMA Strategy',
                                                 'Buy and Hold']))
    )
    # Line chart for CMA Strategy
    cma_strategy_line = base.mark_line().encode(
        y=alt.Y('CMA Strategy:Q', axis=alt.Axis(title='Portfolio value in $')),
        color=alt.value('#4c78a8')
    )
    
    # Line chart for Buy and Hold strategy, with customizations for dashed line and shading
    buy_and_hold_line = base.mark_line(strokeDash=[5,5], opacity=0.5).encode(
        y=alt.Y('Buy and Hold:Q'),
        color=alt.value('grey')
    )

    # Points for buy/sell signals
    buy_signals = base.mark_point(shape='triangle-up', size=100, color='green', strokeWidth=6).encode(
        y='Buy and Hold:Q'
    ).transform_filter(
        alt.datum.Position == 1
    )

    sell_signals = base.mark_point(shape='triangle-down', size=100, color='red', strokeWidth=6).encode(
        y='Buy and Hold:Q'
    ).transform_filter(
        alt.datum.Position == -1
    )

    # Combine all elements into a single chart
    final_chart = (legend_2 + cma_strategy_line + buy_and_hold_line + buy_signals + sell_signals).properties(
        title=f'Buy and Hold vs. Crossover Strategy - {stock}'
    ).interactive()

    # Display chart in Streamlit
    st.altair_chart(final_chart, use_container_width=True)

    # Final value of the portfolio
    final_crossover_total = positions['cma_strategy'][-1]
    final_buy_hold_total = positions['holding_real'][-1]

    # Total returns
    total_return_crossover = final_crossover_total - initial_capital
    total_return_buy_hold = final_buy_hold_total - initial_capital
    total_return_crossover_pct = total_return_crossover/initial_capital*100
    total_return_buy_hold_pct = total_return_buy_hold/initial_capital*100

    # Mean
    mean_crossover = np.mean(stock_data['CMA Strategy Returns'])*100
    mean_buy_hold = np.mean(stock_data['Daily Returns'])*100

    # Standard deviation
    std_crossover = np.std(stock_data['CMA Strategy Returns'])*100
    std_buy_hold = np.std(stock_data['Daily Returns'])*100

    # Calculate volatility for both strategies
    volatility_crossover = stock_data['CMA Strategy Returns'].std() * np.sqrt(252)  # Annualized Volatility
    volatility_buy_hold = stock_data['Daily Returns'].std() * np.sqrt(252)  # Annualized Volatility

    # Sharpe ratio
    sharpe_crossover = mean_crossover / std_crossover
    sharpe_buy_hold = mean_buy_hold / std_buy_hold

    # Maximum drawdown
    max_drawdown_crossover = (stock_data['Cumulative CMA Strategy Returns'] - 
                                    stock_data['Cumulative CMA Strategy Returns'].cummax()).min()
    max_drawdown_buy_hold = (stock_data['Cumulative Buy and Hold Strategy Returns'] - 
                                   stock_data['Cumulative Buy and Hold Strategy Returns'].cummax()).min()

    # Format the returns as strings with proper formatting for readability
    formatted_final_crossover_total = "${:,.2f}".format(final_crossover_total)
    formatted_final_buy_hold_total = "${:,.2f}".format(final_buy_hold_total)
    formatted_total_return_crossover = "${:,.2f}".format(abs(total_return_crossover))
    formatted_total_return_buy_hold = "${:,.2f}".format(abs(total_return_buy_hold))
    formatted_total_return_crossover_pct = "{:.2f}%".format(total_return_crossover_pct)
    formatted_total_return_buy_hold_pct = "{:.2f}%".format(total_return_buy_hold_pct)
    formatet_volatility_crossover = "{:.2f}%".format(volatility_crossover)
    formatet_volatility_buy_hold = "{:.2f}%".format(volatility_buy_hold)
    formatted_sharpe_crossover = "{:.4f}".format(sharpe_crossover)
    formatted_sharpe_buy_hold = "{:.4f}".format(sharpe_buy_hold)
    formatetd_mean_crossover = "{:.4f}%".format(mean_crossover)
    formatetd_mean_buy_hold = "{:.4f}%".format(mean_buy_hold)

    # Create a data frame with this information
    data = {'Crossover Moving Average': [formatted_final_crossover_total,
                                        formatted_total_return_crossover,
                                        formatted_total_return_crossover_pct,
                                        formatetd_mean_crossover, 
                                        formatet_volatility_crossover,
                                        formatted_sharpe_crossover
                                        ],
            'Buy and Hold': [formatted_final_buy_hold_total,
                             formatted_total_return_buy_hold,
                             formatted_total_return_buy_hold_pct,
                             formatetd_mean_buy_hold, 
                             formatet_volatility_buy_hold,
                             formatted_sharpe_buy_hold
                             ]}

    # Create a DataFrame
    df = pd.DataFrame(data, index=['Final Value', 'Total return', 'Total return', 
                                   'Mean daily return', 'Volatility',  'Sharpe ratio' 
                                   ])

    # Return Analysis
    st.write(f"### Returns")
    
    # Determine the performance and positivity/negativity of returns for clearer message construction
    crossover_performance = "outperformed" if total_return_crossover >= 0 else "underperformed"
    buy_hold_performance = "outperformed" if total_return_buy_hold >= 0 else "underperformed"
 
    if total_return_crossover > total_return_buy_hold:
        st.write(f"""
        - The **CMA strategy** has **{crossover_performance}**, yielding a {"positive" if total_return_crossover >=0 else "negative"} return of {formatted_total_return_crossover} ({formatted_total_return_crossover_pct}), which is higher than the Buy and Hold strategy.
        - The **Buy and Hold strategy** has **{buy_hold_performance}**, with a {"positive" if total_return_buy_hold >=0 else "negative"} return of {formatted_total_return_buy_hold} ({formatted_total_return_buy_hold_pct}).

        This analysis suggests that the CMA strategy might be more suited for investors looking for higher returns in the analyzed period. However, investment decisions should also consider other factors such as risk tolerance, investment goals, and market conditions.
        """)
    else:
        st.write(f"""
        **Performance Summary:**

        - The **Buy and Hold strategy** has **{buy_hold_performance}**, yielding a return of {formatted_total_return_buy_hold} ({formatted_total_return_buy_hold_pct}), which is higher than the CMA strategy.
        - The **CMA strategy** has **{crossover_performance}**, with a return of {formatted_total_return_crossover} ({formatted_total_return_crossover_pct}).

        This indicates that the Buy and Hold strategy might be more favorable for investors seeking steadier gains over the analyzed period, emphasizing the importance of aligning strategy choice with individual investment objectives and market outlook.
        """)

    # Visualization (Optional) - Histogram of Daily Returns for Both Strategies
    # Preparing data for visualization
    visualization_data = pd.DataFrame({
        "CMA Strategy Returns": stock_data['CMA Strategy Returns'].dropna(),
        "Buy and Hold Returns": stock_data['Daily Returns'].dropna()
    })

    # Plotting
    st.write("### Return Distribution")
    st.altair_chart(alt.Chart(visualization_data.melt(), width=600).mark_bar(opacity=0.7, binSpacing=0).encode(
        x=alt.X("value:Q", bin=alt.Bin(maxbins=50), title="Daily Returns"),
        y=alt.Y('count()', stack=None, title="Frequency"),
        color=alt.Color('variable:N', legend=alt.Legend(title="Strategy")),
        tooltip=[alt.Tooltip('count()', title='Frequency'), alt.Tooltip('value:Q', title='Daily Return', format='.4%')]
    ).properties(title="Distribution of Daily Returns for Each Strategy").interactive(), use_container_width=True)

   # Sharpe Ratio Analysis
    st.write(f"### Sharpe Ratio")
    st.write("""
    The Sharpe Ratio measures the performance of an investment compared to a risk-free asset, after adjusting for its risk. 
    It is a useful metric for comparing the risk-adjusted performance of different investment strategies.
    """)
    if sharpe_crossover > sharpe_buy_hold:
        st.write(f"""
        - **CMA Strategy Sharpe Ratio**: {formatted_sharpe_crossover}
        - **Buy and Hold Strategy Sharpe Ratio**: {formatted_sharpe_buy_hold}
        
        The Sharpe ratio for the CMA strategy ({formatted_sharpe_crossover}) is higher than that of the buy and hold strategy ({formatted_sharpe_buy_hold}), indicating a better risk-adjusted return.
        This suggests that, relative to the risk taken, the CMA strategy has been more efficient at generating returns.
        
        For investors, a higher Sharpe ratio means that the investment choice has provided better returns for the amount of risk taken. It's an essential factor to consider, especially for those seeking to optimize their investment portfolios' risk/reward balance.
        """)
    else:
        st.write(f"""
        - **CMA Strategy Sharpe Ratio**: {formatted_sharpe_crossover}
        - **Buy and Hold Strategy Sharpe Ratio**: {formatted_sharpe_buy_hold}
        
        The Sharpe ratio for the CMA strategy ({formatted_sharpe_crossover}) is lower than that of the buy and hold strategy ({formatted_sharpe_buy_hold}), indicating a worse risk-adjusted return.
        This means that for the level of risk taken, the buy and hold strategy has provided better returns compared to the CMA strategy.
        
        Investors might prefer a strategy with a higher Sharpe ratio, especially if they are risk-averse or seek to maximize the efficiency of their risk-adjusted returns. However, it's important to consider other factors such as individual risk tolerance, investment horizon, and market conditions when making investment decisions.
        """)

    # Volatility Analysis
    st.write(f"### Volatility Analysis")
    st.write(f"""
             - **CMA Strategy Annualized Volatility**: {volatility_crossover:.2%}
             - **Buy and Hold Strategy Annualized Volatility**: {volatility_buy_hold:.2%}"""
             )
    # Comparative Insight
    if volatility_crossover > volatility_buy_hold:
        st.write("The CMA strategy exhibits higher annualized volatility than the Buy and Hold strategy, indicating a higher risk profile.")
    else:
        st.write("The Buy and Hold strategy exhibits higher annualized volatility than the CMA strategy, indicating a higher risk profile.")

    # Assuming 'stock_data' DataFrame contains the cumulative returns for both strategies

    def calculate_max_drawdown(cumulative_returns):
        """
        Calculate the maximum drawdown for a series of cumulative returns.
        """
        # Calculate the running maximum
        running_max = cumulative_returns.cummax()
        # Calculate drawdown
        drawdown = (cumulative_returns - running_max) / running_max
        # Calculate max drawdown
        max_drawdown = drawdown.min()
        return max_drawdown

    # Calculate Max Drawdown for both strategies
    max_drawdown_cma = calculate_max_drawdown(stock_data['Cumulative CMA Strategy Returns'])
    max_drawdown_buy_hold = calculate_max_drawdown(stock_data['Cumulative Buy and Hold Strategy Returns'])

    # Presenting the insights
    st.write(f"### Drawdown Insights")
    st.write(f"""
            - **Maximum Drawdown for CMA Strategy**: {max_drawdown_cma:.2%}
            - **Maximum Drawdown for Buy and Hold Strategy**: {max_drawdown_buy_hold:.2%}
            """
            )
    if abs(max_drawdown_cma) > abs(max_drawdown_buy_hold):
        st.write("The CMA strategy experienced a larger maximum drawdown than the Buy and Hold strategy, suggesting higher potential risk or larger swings in value.")
    else:
        st.write("The Buy and Hold strategy experienced a larger maximum drawdown than the CMA strategy, indicating it might be subject to higher potential risk or more significant value fluctuations.")
    st.write("A larger maximum drawdown suggests that the strategy may have periods of significant losses, which could test an investor's tolerance for risk and their capacity to withstand downturns without abandoning the strategy.")
    st.write("For investors with a lower risk tolerance, strategies with smaller maximum drawdowns might be more suitable. However, it’s also important to balance the consideration of drawdowns with other factors like overall returns, the Sharpe ratio, and investment goals.")
    st.write("### Drawdown Visualization")

    # Plotting the drawdowns
    
    # Calculate the drawdown for visualization
    stock_data['CMA Drawdown'] = (stock_data['Cumulative CMA Strategy Returns'] - stock_data['Cumulative CMA Strategy Returns'].cummax()) / stock_data['Cumulative CMA Strategy Returns'].cummax()
    stock_data['Buy and Hold Drawdown'] = (stock_data['Cumulative Buy and Hold Strategy Returns'] - stock_data['Cumulative Buy and Hold Strategy Returns'].cummax()) / stock_data['Cumulative Buy and Hold Strategy Returns'].cummax()

    # Prepare the data for visualization
    drawdown_data = stock_data[['CMA Drawdown', 'Buy and Hold Drawdown']].reset_index().melt('Date', var_name='Strategy', value_name='Drawdown')

    # Plotting the drawdowns

    st.altair_chart(alt.Chart(drawdown_data, width=600).mark_line().encode(
        x='Date:T',
        y='Drawdown:Q',
        color='Strategy:N'
    ).properties(title='Drawdowns Over Time for CMA and Buy and Hold Strategies').interactive(), use_container_width=True)
    
    # Summary table
    st.write(f"### Summary table")
    st.write(df)
