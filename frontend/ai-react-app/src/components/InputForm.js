// InputForm.js
import React, { useState } from 'react';

const InputForm = ({ onPredict }) => {
    // const [textInput, setTextInput] = useState('');
    const [teacherVideoFile, setTeacherVideoFile] = useState(null);
    const [studentVideoFile, setStudentVideoFile] = useState(null);


    // Function to handle file input change
    // const handleInputChange = (event) => {
    //     setTextInput(event.target.value);
    // };

    const handleStudentFileChange = (event) => {        
        setStudentVideoFile(event.target.files[0]);
    };
    const handleTeacherFileChange = (event) => {
        setTeacherVideoFile(event.target.files[0]);
    }

    // Function to handle form submission
    // const handleSubmit = (event) => {
    //     event.preventDefault();
    //     if (textInput) {
    //         onPredict(textInput);
    //     } else {
    //         alert('Please enter text for prediction.');
    //     }
    // };

    const handlePredict = () => {
        onPredict(studentVideoFile, teacherVideoFile);
    };

    return (
        <div className="container">
            <h2 className="mt-4">AI Model Input</h2>

            <div className="mb-3">
                <label htmlFor="textInput" className="form-label fs-5 fw-bold">Select a student video:</label>
                {/* <input type="text" className="form-control" id="textInput" value={textInput} onChange={handleInputChange} /> */}
                <input type="file" className="form-control" id="studentVideo" accept=".mp4" onChange={handleStudentFileChange} />
            </div>
            <div className="mb-3">
                <label htmlFor="textInput" className="form-label fs-5 fw-bold">Select a teacher video:</label>
                {/* <input type="text" className="form-control" id="textInput" value={textInput} onChange={handleInputChange} /> */}
                <input type="file" className="form-control" id="teacherVideo" accept=".mp4" onChange={handleTeacherFileChange} />
            </div>
            <button onClick={handlePredict} className="btn btn-primary">Predict</button>

        </div>

    );
};

export default InputForm;
