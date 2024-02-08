// // Import TFJS runtime with side effects.
// import '@tensorflow/tfjs-backend-webgl';
// import * as poseDetection from '@tensorflow-models/pose-detection';
// // import 'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-webgl';
// // import * as poseDetection from 'https://cdn.jsdelivr.net/npm/@tensorflow-models/pose-detection';

/* 
=============
Params
=============
*/

let capture;
let font;

let detector;
let poses;

let angle = 0.0;
let average_latency = 0.0;
let total_time = 0.0;
let frame_count = -25;
let finished_init = false;
/* 
=============
Function
=============
*/

function preload() {
  font = loadFont("./assets/NotoSans-Regular.ttf");
}

function setup() {
  console.log("setup...");
  createCanvas(windowWidth, windowHeight, WEBGL);

  textFont(font);

  // setupCamera();

  // For video file
  capture = createVideo("assets/synced.mp4");
  capture.elt.muted = true;
  capture.loop();
  initModel();

  background(255);
  text("Wait for setup...", 20, 20);
}

async function initModel() {
  const _model = poseDetection.SupportedModels.BlazePose;
  console.log("model:", _model);
  const detectorConfig = {
    runtime: "tfjs", // 'mediapipe', 'tfjs'
    modelType: "lite", // 'lite', 'full', 'heavy'
  };
  detector = await poseDetection.createDetector(_model, detectorConfig);
  finished_init = true;
}

async function getPose() {
  poses = await detector.estimatePoses(capture.elt);
}

function draw() {
  //rectMode(CENTER);
  if (!finished_init) {
    background(255);
    return;
  }
  drawBackground({});

  if (detector && finished_init) {

    // Get current time 
    let start_time = millis();
    getPose();
    // Get end time
    let end_time = millis();
    // Calculate time difference
    let time_diff = end_time - start_time;
    // Round the time difference to 2 decimal places
    time_diff = Math.round(time_diff * 100) / 100;
    if (frame_count < 0) {
      frame_count += 1;
      return;
    }
    total_time += time_diff;
    frame_count += 1;
    // Calculate average latency
    average_latency = total_time / frame_count;
    // Round the average latency to 2 decimal places
    average_latency = Math.round(average_latency * 100) / 100;

    // Draw time difference
    push();
    translate(-windowWidth / 2.0, -windowHeight / 2.0);
    fill(255);
    rect(0, 0, 200, 100);
    fill(0);
    text("Time: " + time_diff + "ms", 20, 20);
    text("Average: " + average_latency + "ms", 20, 40);
    pop();
  }

  drawPoseInfo();
}

function drawBackground({ forWEBGL = true }) {
  // For capture
  push();
  if (forWEBGL) {
    translate(windowWidth / 2.0, -windowHeight / 2.0);
  } else {
    translate(windowWidth, 0);
  }
  scale(-1, 1);
  image(capture, 0, 0, windowWidth, windowHeight);
  pop();

  // For keypoints3D
  push();
  if (forWEBGL) {
    translate(windowWidth / 2.0, -windowHeight / 2.0);
  } else {
    translate(windowWidth, 0);
  }
  noStroke();
  fill(0, 0, 0, 50);
  let size = windowHeight / 4.0;
  rect(-windowWidth, 0, size, size);
  pop();
}

function drawPoseInfo() {
  if (poses && poses.length > 0) {
    angle += 0.01;

    for (var i = 0; i < poses.length; i++) {
      for (var j = 11; j < poses[i].keypoints3D.length; j++) {
        if (poses[i].keypoints3D[j].score > 0.5) {
          let posX =
            -windowWidth / 2.0 +
            (windowWidth -
              poses[i].keypoints[j].x * (windowWidth / capture.width));
          let posY =
            -windowHeight / 2.0 +
            poses[i].keypoints[j].y * (windowHeight / capture.height);

          // Draw circle at keypoint position
          noStroke();
          fill(255, 0, 0, 128);
          circle(posX, posY, 25);

          // Draw keypoint name
          push();
          translate(posX, posY);
          rotateZ(-PI / 4.0);
          fill(255);
          pop();
        }
        // Stop drawing keypoints after 28
        if (j == 28) {
          break;
        }
      }
      // Draw skeleton line between keypoints based on BlazePose skeleton
      drawSkeleton(poses[i].keypoints);
    }
  }
}

function drawSkeleton(keypoints) {
  // Draw skeleton line between keypoints based on BlazePose skeleton
  // https://google.github.io/mediapipe/solutions/pose.html#pose-landmark-model-blazepose-ghum-3d
  // Draw line between left shoulder and right shoulder
  drawLine(keypoints[11], keypoints[12]);
  // Draw line between left shoulder and left hip
  drawLine(keypoints[11], keypoints[23]);
  // Draw line between right shoulder and right hip
  drawLine(keypoints[12], keypoints[24]);
  // Draw line between left hip and right hip
  drawLine(keypoints[23], keypoints[24]);
  // Draw line between left shoulder and left elbow
  drawLine(keypoints[11], keypoints[13]);
  // Draw line between right shoulder and right elbow
  drawLine(keypoints[12], keypoints[14]);
  // Draw line between left elbow and left wrist
  drawLine(keypoints[13], keypoints[15]);
  // Draw line between right elbow and right wrist
  drawLine(keypoints[14], keypoints[16]);
  // Draw line between left hip and left knee
  drawLine(keypoints[23], keypoints[25]);
  // Draw line between right hip and right knee
  drawLine(keypoints[24], keypoints[26]);
  // Draw line between left knee and left ankle
  drawLine(keypoints[25], keypoints[27]);
  // Draw line between right knee and right ankle
  drawLine(keypoints[26], keypoints[28]);
}

// Define drawLine function
function drawLine(keypoint1, keypoint2) {
  let posX1 =
    -windowWidth / 2.0 +
    (windowWidth - keypoint1.x * (windowWidth / capture.width));
  let posY1 =
    -windowHeight / 2.0 + keypoint1.y * (windowHeight / capture.height);
  let posX2 =
    -windowWidth / 2.0 +
    (windowWidth - keypoint2.x * (windowWidth / capture.width));
  let posY2 =
    -windowHeight / 2.0 + keypoint2.y * (windowHeight / capture.height);
  stroke(255, 0, 0, 128);
  strokeWeight(5);
  line(posX1, posY1, posX2, posY2);

}

function mousePressed() {
  noLoop();
}
