import './App.css';
import 'bootstrap/dist/css/bootstrap.min.css';
// App.js
import React, { useRef, useEffect, useState } from 'react';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faPlay, faPause, faStop, faStepBackward, faStepForward } from '@fortawesome/free-solid-svg-icons';

// import { OverlayTrigger } from 'react-bootstrap';
import axios from 'axios';
import InputForm from './components/InputForm';
// import moment from 'moment';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
} from 'chart.js';
import { Line, getElementAtEvent } from 'react-chartjs-2';
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
);

function App() {
  // const [predictions, setPredictions] = useState(null);
  // const [calculation_predictions, setCalculationPredictions] = useState(null);
  const [predictionScoreOption, setPredictionScoreOption] = useState(null);
  const [predictionScoreData, setPredictionScoreData] = useState(null);
  const [teacherAngleOption, setTeacherAngleOption] = useState(null);
  const [teacherAngleData, setTeacherAngleData] = useState(null);
  const [calculationScoreOption, setCalculationScoreOption] = useState(null);
  const [calculationScoreData, setCalculationScoreData] = useState(null);
  // define useState for logs data
  // const [logs, setLogs] = useState(null);
  // const [imageSrc, setImageSrc] = useState([]); // [image1, image2, ...
  const [isChecked, setIsChecked] = useState(false);
  const [selectedModel, setSelectedModel] = useState('model_weight_0'); // Initial selected model
  const [selectedExample, setSelectedExample] = useState('good'); // Initial selected model
  // const [showImage, setShowImage] = useState(false);
  const [predictIds, setPredictIds] = useState([]); // [predict_id1, predict_id2, ...
  const [selectedId, setSelectedId] = useState(null); // Initial selected predict id
  const [leftVideoUrl, setLeftVideoUrl] = useState(null); // Initial selected predict id
  const [rightVideoUrl, setRightVideoUrl] = useState(null); // Initial selected predict id
  // const handleShowImage = () => {
  //   setShowImage(!showImage);
  //   console.log("Mouse enter");
  // };

  const chartScoreRef = useRef();
  const chartAngleRef = useRef();
  const onAngleClick = (event) => {
    const time = getElementAtEvent(chartAngleRef.current, event);
    // seek the video to the time
    if (time.length > 0) {
      const time_value = time[0]['index']
      console.log(time_value);
      document.getElementById('leftVideo').currentTime = (time_value / 60.0);
      // Pause the videos
      document.getElementById('leftVideo').pause();
      document.getElementById('rightVideo').currentTime = (time_value / 60.0);
      document.getElementById('rightVideo').pause();
    }
  }
  const onClick = (event) => {
    const score = getElementAtEvent(chartScoreRef.current, event);
    if (score.length > 0) {
      const section_name = predictionScoreData['labels'][score[0]['index']]
      const predict_id = "predict_" + selectedId;
      console.log(predict_id);
      console.log(section_name);
      const student_type = "student";
      const teacher_type = "teacher";
      setLeftVideoUrl('http://localhost:8000/video/' + student_type + '/' + predict_id + '/' + section_name);
      setRightVideoUrl('http://localhost:8000/video/' + teacher_type + '/' + predict_id + '/' + section_name);

      axios.get('http://localhost:8000/poses/' + predict_id + '/' + section_name)
        .then((response) => {
          // Once you receive the prediction result, update the UI
          console.log(response.data);
          var studentAngleLabels = []
          var studentAngleDataOutput = []
          const studentData = response.data['student']
          console.log(studentData);
          for (let key in studentData) {
            studentAngleLabels.push(key);
            studentAngleDataOutput.push(studentData[key]);
          }

          var datasets = []
          // Create a list 12 colors 
          const colors = ["rgb(255, 99, 132)", "rgb(255, 159, 64)", "rgb(255, 205, 86)",
            "rgb(75, 192, 192)", "rgb(54, 162, 235)", "rgb(153, 102, 255)",
            "rgb(201, 203, 207)", "rgb(255, 99, 132)", "rgb(255, 159, 64)",
            "rgb(255, 205, 86)", "rgb(75, 192, 192)", "rgb(54, 162, 235)"]
          for (let i = 0; i < studentAngleDataOutput.length; i++) {

            datasets.push({
              label: studentAngleLabels[i],
              data: studentAngleDataOutput[i],
              fill: false,
              borderColor: colors[1],
              borderWidth: 1,
              tension: 0.1,
              pointRadius: 2,
              pointHoverRadius: 2,
              hidden: i !== 0,
            })
          }
          // Get the label is list from 0 to 1200
          const labels = Array.from(Array(1200).keys());
          const teacherData = response.data['teacher']
          console.log(teacherData);
          var teacherAngleLabels = []
          var teacherAngleDataOutput = []

          for (let key in teacherData) {
            teacherAngleLabels.push(key);
            teacherAngleDataOutput.push(teacherData[key]);
          }

          const teacherAngleOption = {
            responsive: true,
            plugins: {
              legend: {
                labels: {
                  position: 'bottom'
                }
              },
              title: {
                display: true,
                text: 'Teacher Angle Chart',
              },
            },
          };

          setTeacherAngleOption(teacherAngleOption);
          for (let i = 0; i < teacherAngleDataOutput.length; i++) {
            datasets.push({
              label: teacherAngleLabels[i],
              data: teacherAngleDataOutput[i],
              fill: false,
              borderColor: colors[0],
              borderWidth: 1,
              pointRadius: 2,
              pointHoverRadius: 2,
              tension: 0.1,
              hidden: i !== 0,
            })
          }
          const teacherAngleData = {
            labels: labels,
            datasets: datasets
          };

          setTeacherAngleData(teacherAngleData);
        });
    }
  }

  useEffect(() => {
    axios.get('http://localhost:8000/get_predict_ids/')
      .then((response) => {
        // Once you receive the prediction result, update the UI
        console.log(response.data);
        setPredictIds(response.data);
        setSelectedId(response.data[0]);
      });
  }, []);

  // When setSelectedId is updated, this function will be called
  useEffect(() => {
    if (selectedId) {
      handleButtonClick(selectedId);
    }
  }, [selectedId]);


  const handleDeleteButtonClick = () => {
    // Get current select predict id
    // Replace with your actual API URL
    let predict_id = selectedId;
    if (!predict_id) {
      alert('Please select a predict id.');
      return;
    }

    axios.post(`http://localhost:8000/delete_predicts_by_id?predict_id=${predict_id}`)
      .then(response => {
        console.log(response.data);
        if (response.data === 'success') {
          axios.get('http://localhost:8000/get_predict_ids/')
            .then((response) => {
              // Once you receive the prediction result, update the UI
              console.log(response.data);
              setPredictIds(response.data);
              // setSelectedId(response.data[0]);
              // Select the first predict id
              setSelectedId(response.data[0]);
            });
        } else {
          console.log(response.data);
          alert('The prediction is not deleted successfully.');
        }
      });
  };

  const handlePredictAgainButtonClick = () => {
    // Get current select predict id
    // Replace with your actual API URL
    let predict_id = selectedId;
    if (!predict_id) {
      alert('Please select a predict id.');
      return;
    }

    axios.post(`http://localhost:8000/predict_again?predict_id=${predict_id}`)
      .then(response => {
        console.log(response.data);
        if (response.data === 'success') {
          axios.get('http://localhost:8000/get_predict_ids/')
            .then((response) => {
              // Once you receive the prediction result, update the UI
              console.log(response.data);
              setPredictIds(response.data);
              // setSelectedId(response.data[0]);
              // Select the first predict id
              setSelectedId(predict_id);
            });
        } else {
          console.log(response.data);
          alert('The prediction is not re-predict successfully.');
        }
      }
      );
  };



  const handleButtonClick = (predict_id) => {
    // Replace with your actual API URL
    axios.get(`http://localhost:8000/get_score?predict_id=${predict_id}`)
      .then(response => {
        console.log(response.data);
        if (response.data.score === null) {
          alert('The prediction is not ready yet. Please try again later.');
          return;
        } else {
          var sectionScores = response.data[1];
          // setSelectedId(response.data[0][0][1]);
          console.log(sectionScores);
          var predictionScoreLabels = []
          var predictionScores = []
          var timingCalculateScores = []
          var velocityCalculateScores = []
          var accuracyCalculateScores = []
          var calculateScoreLabels = []

          for (let i = 0; i < sectionScores.length; i++) {
            let section_name = sectionScores[i][2];
            predictionScoreLabels.push(section_name);
            calculateScoreLabels.push(section_name);
            let ai_score = parseFloat(sectionScores[i][3])
            predictionScores.push(ai_score);
            let timing_score = parseFloat(sectionScores[i][4])
            timingCalculateScores.push(timing_score);
            let velocity_score = parseFloat(sectionScores[i][5])
            velocityCalculateScores.push(velocity_score);
            let accuracy_score = parseFloat(sectionScores[i][6])
            accuracyCalculateScores.push(accuracy_score);
          }

          const calculationScoreOption = {
            responsive: true,
            plugins: {
              legend: {
                position: 'top',
              },
            },
            scales: {
              x: {
                title: {
                  display: true,
                  text: 'Section Name'
                },
              },
              y: {
                title: {
                  display: true,
                  text: 'Score'
                },
                min: 0,
                max: 5,
                stepSize: 1,
              }
            }

          };

          const calculationScoreData = {
            labels: calculateScoreLabels,
            datasets: [{
              label: 'Timing Score',
              data: timingCalculateScores,
              fill: false,
              borderColor: 'rgb(255, 99, 132)',
              tension: 0.1
            },
            {
              label: 'Velocity Score',
              data: velocityCalculateScores,
              fill: false,
              borderColor: 'rgb(53, 162, 235)',
              tension: 0.1
            },
            {
              label: 'Accuracy Score',
              data: accuracyCalculateScores,
              fill: false,
              borderColor: 'rgb(162, 53, 235)',
              tension: 0.1
            }]
          };

          setCalculationScoreOption(calculationScoreOption);
          setCalculationScoreData(calculationScoreData);

          const predictionScoreOption = {
            responsive: true,
            plugins: {
              legend: {
                position: 'top',
              },
            },
            scales: {
              x: {
                title: {
                  display: true,
                  text: 'Section Name'
                },
              },
              y: {
                title: {
                  display: true,
                  text: 'Score'
                },
                min: 0,
                max: 5,
                stepSize: 1,
              }
            }
          };

          const predictionScoreData = {
            labels: predictionScoreLabels,
            datasets: [{
              label: 'AI Score',
              data: predictionScores,
              fill: false,
              borderColor: 'rgb(255, 99, 132)',
              tension: 0.1
            }]
          };

          setPredictionScoreOption(predictionScoreOption);
          setPredictionScoreData(predictionScoreData);
        }
      });
  };

  // let currentMoment = null;
  // let totalTimeDiff = 0;

  const handleCheckboxChange = () => {
    setIsChecked(!isChecked); // Toggle the checkbox state
  };

  const handleModelChange = (event) => {
    setSelectedModel(event.target.value);
  };

  const handleExampleChange = (event) => {
    setSelectedExample(event.target.value);
  };

  const handlePredict = (studentData, teacherData) => {
    // currentMoment = null;
    // totalTimeDiff = 0;
    // Reset the prediction result
    // setPredictions(null);
    setPredictionScoreOption(null);
    setPredictionScoreData(null);
    setCalculationScoreOption(null);
    setCalculationScoreData(null);
    if (isChecked) {
      axios({
        method: 'post',
        url: 'http://localhost:8000/predict_skip_upload/',
        data: JSON.stringify({ 'model': selectedModel, 'good_or_bad': selectedExample }),
        headers: {
          'Accept': 'application/json',
          'Content-Type': 'application/json'
        },
      })
        .then((response) => {
          // Once you receive the prediction result, update the UI
          console.log(response.data.predictions);
          // setPredictions(response.data.predictions);
          // setCalculationPredictions(response.data.calculation_predictions);
          // setLogs(response.data.logs);
        })
        .catch((error) => {
          console.error('Prediction error:', error);
        });
    } else {
      if (!studentData) {
        alert('Please select a student video file.');
        return;
      }
      if (!teacherData) {
        alert('Please select a teacher video file.');
        return;
      }
      const formData = new FormData();
      formData.append('student', studentData);
      formData.append('teacher', teacherData);
      formData.append('model', selectedModel);

      axios
        .post('http://localhost:8000/predict/', formData, {
          headers: {
            'Content-Type': 'multipart/form-data', // Set the content type for file upload
          },
        })
        .then((response) => {
          // Once you receive the prediction result, update the UI
          console.log(response.data.predictions);
          // setPredictions(response.data.predictions);
          // setCalculationPredictions(response.data.calculation_predictions);
          var predictionScores = []
          var predictionScoreLabels = []

          for (let i = 0; i < response.data.predictions.length - 1; i++) {
            let text = response.data.predictions[i];
            let label = text.split(":")[0]
            let score = parseFloat(text.split(":")[1].split("/")[0])
            predictionScores.push(score);
            predictionScoreLabels.push(label);
          }

          const predictionScoreOption = {
            responsive: true,
            plugins: {
              legend: {
                position: 'top',
              },
              title: {
                display: true,
                text: 'Prediction Chart',
              },
            },
          };
          setPredictionScoreOption(predictionScoreOption);

          const predictionScoreData = {
            labels: predictionScoreLabels,
            datasets: [{
              label: 'AI Score',
              data: predictionScores,
              fill: false,
              borderColor: 'rgb(255, 99, 132)',
              tension: 0.1
            }]
          };

          setPredictionScoreData(predictionScoreData);

          var timingCalculateScores = []
          var velocityCalculateScores = []
          var accuracyCalculateScores = []
          var calculateScoreLabels = []

          for (let i = 0; i < response.data.calculation_predictions.length - 1; i++) {
            let text = response.data.calculation_predictions[i];
            let label = text.split(",")[0]
            let timing_score = parseFloat(text.split(",")[1].split(":")[1].split("/")[0])
            let velocity_score = parseFloat(text.split(",")[2].split(":")[1].split("/")[0])
            let accuracy_score = parseFloat(text.split(",")[3].split(":")[1].split("/")[0])
            timingCalculateScores.push(timing_score);
            velocityCalculateScores.push(velocity_score);
            accuracyCalculateScores.push(accuracy_score);
            calculateScoreLabels.push(label);
          }

          const calculationScoreOption = {
            responsive: true,
            plugins: {
              legend: {
                position: 'top',
              },
              title: {
                display: true,
                text: 'Calculation Chart',
              },
            },
          };
          setCalculationScoreOption(calculationScoreOption);

          const calculationScoreData = {
            labels: calculateScoreLabels,
            datasets: [{
              label: 'Timing Score',
              data: timingCalculateScores,
              fill: false,
              borderColor: 'rgb(255, 99, 132)',
              tension: 0.1
            },
            {
              label: 'Velocity Score',
              data: velocityCalculateScores,
              fill: false,
              borderColor: 'rgb(53, 162, 235)',
              tension: 0.1
            },
            {
              label: 'Accuracy Score',
              data: accuracyCalculateScores,
              fill: false,
              borderColor: 'rgb(162, 53, 235)',
              tension: 0.1
            }]
          };

          setCalculationScoreData(calculationScoreData);

          axios.get('http://localhost:8000/get_predict_ids/')
            .then((response) => {
              // Once you receive the prediction result, update the UI
              console.log(response.data);
              setPredictIds(response.data);
              // Set the last predict id as the selected predict id
              setSelectedId(response.data[response.data.length - 1]);
            });
          // setLogs(response.data.logs);
        })
        .catch((error) => {
          console.error('Prediction error:', error);
        });
    }
  };

  const handlePlay = () => {
    // Add logic for play button
    console.log('Play button clicked');
    // Play 2 video player
    document.getElementById('leftVideo').play();
    document.getElementById('rightVideo').play();
  };

  const handlePause = () => {
    // Add logic for pause button
    console.log('Pause button clicked');
    // Pause 2 video player
    document.getElementById('leftVideo').pause();
    document.getElementById('rightVideo').pause();
  };

  const handleStop = () => {
    // Add logic for stop button
    console.log('Stop button clicked');
    // Stop 2 video player
    document.getElementById('leftVideo').pause();
    document.getElementById('leftVideo').currentTime = 0;
    document.getElementById('rightVideo').pause();
    document.getElementById('rightVideo').currentTime = 0;
  };

  const handleStepBackward = () => {
    // Add logic for stop button
    console.log('StepBackward button clicked');
    // Stop 2 video player
    //document.getElementById('leftVideo').pause();
    document.getElementById('leftVideo').currentTime -= (1 / 60.0);
    //document.getElementById('rightVideo').pause();
    document.getElementById('rightVideo').currentTime -= (1 / 60.0);
  };

  const handleStepForward = () => {
    // Add logic for stop button
    console.log('StepForward button clicked');
    // Stop 2 video player
    //document.getElementById('leftVideo').pause();
    document.getElementById('leftVideo').currentTime += (1 / 60.0);
    //document.getElementById('rightVideo').pause();
    document.getElementById('rightVideo').currentTime += (1 / 60.0);
  };

  // Function to convert array buffer to base64
  // const arrayBufferToBase64 = (buffer) => {
  //   let binary = '';
  //   const bytes = new Uint8Array(buffer);
  //   const len = bytes.byteLength;
  //   for (let i = 0; i < len; i++) {
  //     binary += String.fromCharCode(bytes[i]);
  //   }
  //   return btoa(binary);
  // };

  return (
    <div className="App">
      <h1 className="display-4">AI Dance Evaluation App</h1>
      <div className="container mt-4">
        <div className="form-check form-check-inline">
          <input
            className="form-check-input"
            type="checkbox"
            checked={isChecked}
            onChange={handleCheckboxChange}
            id="showTextCheckbox"
          />
          <label className="form-check-label" htmlFor="showTextCheckbox">
            Skip upload videos
          </label>
        </div>
        {isChecked && (
          <div className="alert alert-success mt-2" role="alert">
            The videos will be skipped and the default videos will be used.
          </div>
        )}
      </div>
      {isChecked && (
        <div className="container mt-4">
          <h4>Choose Good or Bad example</h4>
          <div className="input-group">
            <select
              className="form-select custom-select"
              value={selectedExample}
              onChange={handleExampleChange}
            >
              <option value="good">Good Example</option>
              <option value="bad">Bad Example</option>
            </select>
          </div>
        </div>
      )}
      <div className="container mt-4">
        <h4>Choose AI Weight Model</h4>
        <div className="input-group">
          <select
            className="form-select custom-select"
            value={selectedModel}
            onChange={handleModelChange}
          >
            <option value="model_weight_0">model_weight_0</option>
            <option value="model_weight_1">model_weight_1</option>
            <option value="model_weight_2">model_weight_2</option>
            <option value="model_weight_3">model_weight_3</option>
          </select>
        </div>
        <p className="mt-2">Selected Model: {selectedModel}</p>
      </div>
      {!isChecked && (
        <InputForm onPredict={handlePredict} />
      )}
      {isChecked && (
        <button onClick={handlePredict} className="btn btn-primary">Predict</button>
      )}

      <div className="container mt-4">
        <h5>Choose Predict ID</h5>
        <div className="input-group">
          <select className="form-select custom-select" onChange={e => setSelectedId(e.target.value)}>
            {predictIds.map(id => (
              <option key={id} value={id}>{id}</option>
            ))}
          </select>
          <button className="btn btn-primary" onClick={handleDeleteButtonClick}>Delete</button>
          <button className="btn btn-danger" onClick={handlePredictAgainButtonClick}>Predict Again</button>
        </div>
      </div>
      {(predictionScoreData) && (
        <div className="mt-4">
          <div className="container mt-4">
            <div className="row">
              <div className="col-6">
                <h4>Multi Factors Evaluation</h4>
                <ul className="list-group">
                  <Line ref={chartScoreRef} onClick={onClick} options={calculationScoreOption} data={calculationScoreData} />
                </ul>
              </div>
              <div className="col-6">
                <h4>Artistic Evaluation</h4>
                <ul className="list-group">
                  <Line options={predictionScoreOption} data={predictionScoreData} />
                </ul>
              </div>
            </div>
          </div>
        </div>)}      
      {(teacherAngleData) && (
        <div className="mt-4">
          <div className="container mt-4">
            <div className="row">
              <div>
                <h4>Teacher Angle Data</h4>
                <ul className="list-group">
                  <Line ref={chartAngleRef} onClick={onAngleClick} options={teacherAngleOption} data={teacherAngleData} />
                </ul>
              </div>
            </div>
          </div>
        </div>)}
      {(leftVideoUrl && rightVideoUrl) && (
        <div className="mt-4">
          <div className="container mt-4">
            <div>
              <button className="btn btn-info mx-2" onClick={handleStepBackward}>
                <FontAwesomeIcon icon={faStepBackward} />
              </button>
              <button className="btn btn-primary mx-2" onClick={handlePlay}>
                <FontAwesomeIcon icon={faPlay} />
              </button>
              <button className="btn btn-warning mx-2" onClick={handlePause}>
                <FontAwesomeIcon icon={faPause} />
              </button>
              <button className="btn btn-danger mx-2" onClick={handleStop}>
                <FontAwesomeIcon icon={faStop} />
              </button>
              <button className="btn btn-info mx-2" onClick={handleStepForward}>
                <FontAwesomeIcon icon={faStepForward} />
              </button>
            </div>
            <div className="row">
              <div className="col-6">
                <h2>Left Video</h2>
                <video id="leftVideo" controls width={600} height={400} autoPlay={true} src={leftVideoUrl} type="video/mp4">
                  Your browser does not support the video tag.
                </video>
              </div>
              <div className="col-6">
                <h2>Right Video</h2>
                <video id="rightVideo" controls width={600} height={400} autoPlay={true} src={rightVideoUrl} type="video/mp4">
                  Your browser does not support the video tag.
                </video>
              </div>
            </div>
          </div>
        </div>)}
    </div>
  );
}

export default App;
