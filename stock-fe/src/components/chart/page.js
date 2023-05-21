import ChartComponent from '../ChartComponent'
import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';

export default async function ChartPage() {
 

  const handleButtonClick = () => {

    const apiUrl = `YOUR_API_ENDPOINT?selectedOption=${selectedOption}&inputValue=${inputValue}`;
    fetch(apiUrl)
      .then((response) => response.json())
      .then((data) => {
        // Handle the API response data
        console.log(data);
      })
      .catch((error) => {
        // Handle errors
        console.error(error);
      });
  };
  async function getData() {
    const response = await axios.get('http://localhost:8000/get-vn30-history/');
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

    console.log(formattedData);
    // setData(formattedData);
    return formattedData;
  }

  return (
    <main className="flex min-h-screen flex-col items-center justify-between md:p-24">

      {/* Chart Area */}
      <div className='z-10 w-full md:max-w-8xl max-h-[80%] items-center justify-between font-mono text-sm lg:flex'>
        <ChartComponent className="w-full h-full" data={data}
        ></ChartComponent>
      </div>

    </main>
  )
}
