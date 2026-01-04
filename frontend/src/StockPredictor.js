import React, { useState } from 'react';
import axios from 'axios';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import './StockPredictor.css';

const StockPredictor = () => {
    const [ticker, setTicker] = useState('NVDA');
    const [prediction, setPrediction] = useState(null);
    const [historicalData, setHistoricalData] = useState([]);
    const [error, setError] = useState('');
    const [priceGoesUp, setPriceGoesUp] = useState(true);
    const [percentageChange, setPercentageChange] = useState(0);

    const fetchPrediction = async () => {
        try {
            const predictionResponse = await axios.get(`http://localhost:8000/predict/${ticker}`);
            const predictionData = predictionResponse.data;
            setPrediction(predictionData.predicted_price);
            fetchHistoricalData(ticker, predictionData);
        } catch (err) {
            setError('Error fetching prediction. Please check the ticker and try again.');
            console.error(err);
        }
    };

    const fetchHistoricalData = async (selectedTicker, predictionData) => {
        try {
            const historicalResponse = await axios.get(`http://localhost:8000/historical/${selectedTicker}`);
            const historical = historicalResponse.data;

            const newPoint = {
                Date: predictionData.prediction_date,
                predicted: predictionData.predicted_price,
            };

            // To connect the prediction line, the last historical point also needs a 'predicted' value.
            // This will be the starting point for the prediction line.
            if (historical.length > 0) {
                const lastHistoricalPoint = historical[historical.length - 1];
                const change = ((predictionData.predicted_price - lastHistoricalPoint.Close) / lastHistoricalPoint.Close) * 100;
                setPercentageChange(change);
                setPriceGoesUp(predictionData.predicted_price >= lastHistoricalPoint.Close);
                const updatedHistoricalData = historical.map((d, index) => {
                    if (index === historical.length - 1) {
                        return { ...d, predicted: d.Close };
                    }
                    return d;
                });
                setHistoricalData([...updatedHistoricalData, newPoint]);
            } else {
                setHistoricalData([newPoint]);
            }

        } catch (err) {
            setError('Error fetching historical data.');
            console.error(err);
        }
    };

    const handleTickerChange = (event) => {
        setTicker(event.target.value.toUpperCase());
    };

    const handleSubmit = (event) => {
        event.preventDefault();
        setError('');
        setPrediction(null);
        setHistoricalData([]);
        fetchPrediction();
    };

    return (
        <div className="stock-predictor-container">
            <div className="header">
                <h1>Stock Price Predictor</h1>
            </div>
            <form onSubmit={handleSubmit} className="form-container">
                <input
                    type="text"
                    value={ticker}
                    onChange={handleTickerChange}
                    placeholder="Enter Stock Ticker (e.g., NVDA)"
                    className="ticker-input"
                />
                <button type="submit" className="predict-button">
                    Predict
                </button>
            </form>

            {error && <p className="error-message">{error}</p>}

            {prediction && (
                <div className="prediction-result">
                    <h2>
                        Predicted Price for {ticker}: 
                        <span className={`predicted-price ${priceGoesUp ? '' : 'down'}`}>
                            ${prediction.toFixed(2)}
                        </span>
                        <span className={`percentage-change ${priceGoesUp ? 'up' : 'down'}`}>
                            {priceGoesUp ? '▲' : '▼'} {Math.abs(percentageChange).toFixed(2)}%
                        </span>
                    </h2>
                </div>
            )}

            {historicalData.length > 0 && (
                <div className="chart-container">
                    <h3>Historical Data & Prediction</h3>
                    <ResponsiveContainer width="100%" height={400}>
                        <LineChart data={historicalData}>
                            <CartesianGrid strokeDasharray="3 3" />
                            <XAxis dataKey="Date" allowDuplicatedCategory={false} />
                            <YAxis />
                            <Tooltip 
                                contentStyle={{
                                    backgroundColor: 'rgba(30, 30, 33, 0.8)',
                                    borderColor: '#c3073f'
                                }}
                                itemStyle={{ color: '#f5f5f5' }}
                                labelStyle={{ color: '#a0a0a0' }}
                            />
                            <Legend />
                            <Line type="monotone" dataKey="Close" stroke="#6bbfd8" name="Past Price" dot={false} />
                            <Line 
                                type="monotone" 
                                dataKey="predicted" 
                                stroke={priceGoesUp ? '#28a745' : '#c3073f'} 
                                name="Predicted Price" 
                                connectNulls 
                                dot={(props) => {
                                    const { cx, cy, payload, index } = props;
                                    if (index === historicalData.length - 1) {
                                        const color = priceGoesUp ? '#28a745' : '#c3073f';
                                        return <circle cx={cx} cy={cy} r={5} fill={color} />;
                                    }
                                    return null;
                                }}
                            />
                        </LineChart>
                    </ResponsiveContainer>
                </div>
            )}
        </div>
    );
};

export default StockPredictor;
