import React from 'react';
import { Player, ControlBar } from 'video-react';

import 'bootstrap/dist/css/bootstrap.min.css';

const VideoPage = ({ leftVideoUrl, rightVideoUrl }) => {
    
    return (
        <div className="container mt-5">
            <div className="row">
                {/* Left Video */}
                <div className="col-md-6">
                    <h2>Left Video</h2>
                    <Player fluid={false} width={700} height={600} autoPlay>
                        <source src={leftVideoUrl} />
                        <ControlBar autoHide={false} />
                    </Player>                    
                </div>

                {/* Right Video */}
                <div className="col-md-6">
                    <h2>Right Video</h2>
                    <video controls width={700} height={600} src={rightVideoUrl} type="video/mp4">
                        Your browser does not support the video tag.
                    </video>
                </div>
            </div>
        </div>
    );
};

export default VideoPage;
