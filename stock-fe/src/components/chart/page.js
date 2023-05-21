import ChartComponent from '../ChartComponent'
import React, { useState, useEffect } from 'react';
import axios from 'axios';

export default function ChartPage() {
  const [selectedOption, setSelectedOption] = useState('option1');
  const [inputValue, setInputValue] =useState('');
  const [data, setData] =useState([]);
  const [predictData, setPredictData] = useState([]);
  useEffect(() => {
    getData();
  }, []);
  const handleOptionChange = (event) => {
    setSelectedOption(event.target.value);
  };

  const handleInputChange = (event) => {
    setInputValue(event.target.value);
  };

  const handleButtonClick = async () => {
    try {
      const response = await axios.get(`YOUR_API_ENDPOINT?selectedOption=${selectedOption}&inputValue=${inputValue}`);
      setPredictData(response.data);
    } catch(err) { 
      console.log("error", err);
    }
  };

  async function getData() {
    const response = await axios.get('https://v30.ftisu.vn/get-vn30-history/', { withCredentials: true });
    const data = response.data;
    const formattedData = data.map(element => ({
      time: new Date(element['Date']).toISOString().substring(0, 10),
      open: element['Open'],
      close: element['Price'],
      high: element['High'],
      low: element['Low'],
      volume: element['Vol'],
      change: element['Change'],
    })).sort((a, b) => Number(new Date(a['time'])) - Number(new Date(b['time'])));
    setData(formattedData);
  };

  return (
    <main className="flex min-h-screen flex-col items-center justify-between md:p-24">

      {/* Chart Area */}
      <div className='z-10 w-full md:max-w-8xl max-h-[80%] items-center justify-between font-mono text-sm lg:flex'>
        <ChartComponent className="w-full h-full" />
      </div>
      {/* Radio Selections */}
      <div>
          <label>
            <input
              type="radio"
              value="option1"
              checked={selectedOption === 'option1'}
              onChange={handleOptionChange}
            />
            Option 1
          </label>
          <label>
            <input
              type="radio"
              value="option2"
              checked={selectedOption === 'option2'}
              onChange={handleOptionChange}
            />
            Option 2
          </label>
          {/* Add more radio options as needed */}
        </div>

        {/* Input */}
        <div>
          <input
            type="text"
            value={inputValue}
            onChange={handleInputChange}
            placeholder="Enter a value"
          />
        </div>

        {/* Button */}
        <div>
          <button onClick={handleButtonClick} predictdata={predictData}>Execute API</button>
        </div>
    </main>
  )
}
