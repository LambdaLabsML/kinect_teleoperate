// For kinect camera driver
#include <k4arecord/playback.h>
#include <k4a/k4a.h>

// For kinect body tracking and render
#include <k4abt.h>
#include <BodyTrackingHelpers.h>
#include <Utilities.h>
#include <Window3dWrapper.h>

// STL
#include <chrono>
#include <thread>
#include <mutex>
#include <array>
#include <iostream>
#include <vector>
#include <string>

// For mujoco render
#include <mujoco/mujoco.h>
#include <GLFW/glfw3.h>

// Unitree SDK includes for real robot control
#include <unitree/robot/channel/channel_factory.hpp>
#include <unitree/robot/channel/channel_publisher.hpp>
#include <unitree/robot/channel/channel_subscriber.hpp>
#include <unitree/idl/hg/LowCmd_.hpp>
#include <unitree/idl/hg/LowState_.hpp>
#include <unitree/idl/hg/IMUState_.hpp>   // Added for IMUState messages (torso IMU)
#include <unitree/robot/b2/motion_switcher/motion_switcher_client.hpp>
#include <unitree/common/time/time_tool.hpp>
#include <unitree/common/thread/thread.hpp>
// Additional includes from G1 example
#include "gamepad.hpp"  // Added for Gamepad support used in G1Example
// For math tool
#include "math_tool.hpp"
// For start or end state detect
#include "StartEndPoseDetector.hpp"
// For retargeting function from the skeleton joint angles to the robot motor joint angles.
#include "jointRetargeting.hpp"

using namespace std::chrono;
using namespace unitree::common;
using namespace unitree::robot;

bool s_isRunning = true;
bool wakeup = false;

#define Control_G1 true
#define Control_H1 false
#define Real_Control true    // control real unitree robot in reality
#define Enable_Torso false    // enable torso rotation angle mapping. Test function, open with caution!
#define Enable_Hand  false    // enable hand opening and closing status detection. Test function, open with caution!
#define EchoFrequency false   // Whether to display the running frequency of each thread

#if Real_Control
// SDK include files for unitree robot real control
// Please refer to https://github.com/unitreerobotics/unitree_sdk2
#endif


// Define a buffer utility and data structures from G1Example for real robot state/command management
template <typename T>
class DataBuffer {
 public:
   void SetData(const T &newData) {
     std::unique_lock<std::shared_mutex> lock(mutex);
     data = std::make_shared<T>(newData);
   }

   std::shared_ptr<const T> GetData() {
     std::shared_lock<std::shared_mutex> lock(mutex);
     return data ? data : nullptr;
   }

   void Clear() {
     std::unique_lock<std::shared_mutex> lock(mutex);
     data = nullptr;
   }

 private:
   std::shared_ptr<T> data;
   std::shared_mutex mutex;
};

// G1 robot joint count
const int G1_NUM_MOTOR = 29;

// Structures for IMU and motor data (used by G1Example for real robot control)
struct ImuState {
   std::array<float, 3> rpy = {};    // roll, pitch, yaw
   std::array<float, 3> omega = {};  // angular velocities
};
struct MotorCommand {
   std::array<float, G1_NUM_MOTOR> q_target = {};
   std::array<float, G1_NUM_MOTOR> dq_target = {};
   std::array<float, G1_NUM_MOTOR> kp = {};
   std::array<float, G1_NUM_MOTOR> kd = {};
   std::array<float, G1_NUM_MOTOR> tau_ff = {};
};
struct MotorState {
   std::array<float, G1_NUM_MOTOR> q = {};
   std::array<float, G1_NUM_MOTOR> dq = {};
};

// Stiffness (Kp) and Damping (Kd) constants for all G1 joints (from Unitree G1 specs)
std::array<float, G1_NUM_MOTOR> Kp = {
    60, 60, 60, 100, 40, 40,      // legs
    60, 60, 60, 100, 40, 40,      // legs
    60, 40, 40,                   // waist
    40, 40, 40, 40, 40, 40, 40,   // arms
    40, 40, 40, 40, 40, 40, 40    // arms
};
std::array<float, G1_NUM_MOTOR> Kd = {
    1, 1, 1, 2, 1, 1,    // legs
    1, 1, 1, 2, 1, 1,    // legs
    1, 1, 1,             // waist
    1, 1, 1, 1, 1, 1, 1, // arms
    1, 1, 1, 1, 1, 1, 1  // arms
};

// Modes for ankle control (parallel vs. series) as defined in the G1 example
enum class Mode {
   PR = 0,  // Series control for Pitch/Roll joints (PR mode)
   AB = 1   // Parallel control for A/B joints (AB mode)
};

// Joint index mapping for G1 (29 DOF). Important ankle indices are defined for clarity.
enum G1JointIndex {
   LeftHipPitch = 0,
   LeftHipRoll  = 1,
   LeftHipYaw   = 2,
   LeftKnee     = 3,
   LeftAnklePitch = 4,   // Also referred to as LeftAnkleB in AB mode
   LeftAnkleB   = 4,
   LeftAnkleRoll = 5,    // Also referred to as LeftAnkleA in AB mode
   LeftAnkleA   = 5,
   RightHipPitch = 6,
   RightHipRoll  = 7,
   RightHipYaw   = 8,
   RightKnee     = 9,
   RightAnklePitch = 10, // Also referred to as RightAnkleB
   RightAnkleB   = 10,
   RightAnkleRoll = 11,  // Also referred to as RightAnkleA
   RightAnkleA   = 11,
   WaistYaw    = 12,
   WaistRoll   = 13,  // (Unused in some models)
   WaistA      = 13,  // (Unused in some models)
   WaistPitch  = 14,
   WaistB      = 14,
   LeftShoulderPitch = 15,
   LeftShoulderRoll  = 16,
   LeftShoulderYaw   = 17,
   LeftElbow         = 18,
   LeftWristRoll     = 19,
   LeftWristPitch    = 20,  // (Unused for G1 23-DoF)
   LeftWristYaw      = 21,  // (Unused for G1 23-DoF)
   RightShoulderPitch = 22,
   RightShoulderRoll  = 23,
   RightShoulderYaw   = 24,
   RightElbow         = 25,
   RightWristRoll     = 26,
   RightWristPitch    = 27, // (Unused for G1 23-DoF)
   RightWristYaw      = 28  // (Unused for G1 23-DoF)
};

// Utility function for CRC calculation (to validate data integrity of LowState and LowCmd messages)
inline uint32_t Crc32Core(uint32_t *ptr, uint32_t len) {
   uint32_t CRC32 = 0xFFFFFFFF;
   const uint32_t dwPolynomial = 0x04c11db7;
   for (uint32_t i = 0; i < len; i++) {
       uint32_t data = ptr[i];
       uint32_t xbit = 1 << 31;
       for (uint32_t bits = 0; bits < 32; bits++) {
           if (CRC32 & 0x80000000) {
               CRC32 <<= 1;
               CRC32 ^= dwPolynomial;
           } else {
               CRC32 <<= 1;
           }
           if (data & xbit) {
               CRC32 ^= dwPolynomial;
           }
           xbit >>= 1;
       }
   }
   return CRC32;
}

// kinect body tracking skeleton joint angle
// reference: https://learn.microsoft.com/en-us/azure/kinect-dk/body-joints
// ':=' means that the item on the left hand side is being defined to be what is on the right hand side.
// sc:=spine chest, ls:=left shoulder, le:=left elbow, rs:=right shoulder, re:=right elbow, lh:=left hand, rh:=right hand
// _r:=roll, _p:=pitch, _y:=yaw, _a:=angle
static float sc_r, sc_p, sc_y, ls_r, ls_p, ls_y, le_r, le_p, le_y, rs_r, rs_p, rs_y, re_r, re_p, re_y, lh_a, rh_a;

struct hardware_control_signal {
    double left_shoulder_roll = 0.0;
    double left_shoulder_pitch = 0.0;
    double left_shoulder_yaw = 0.0;
    double right_shoulder_roll = 0.0;
    double right_shoulder_pitch = 0.0;
    double right_shoulder_yaw = 0.0;
    double left_elbow_yaw = 0.0;
    double right_elbow_yaw = 0.0;
};


// For control real robot G1
#if Control_G1
hardware_control_signal G1_hardware_signal;
#endif

// For control real robot H1
#if Control_H1
hardware_control_signal H1_hardware_signal;
#endif

// G1Example class from the ankle swing example, integrated for real robot control
class G1Example {
 private:
   // Timing and mode variables
   double time_;
   double control_dt_;   // control loop period (2ms)
   double duration_;     // stage duration (3s per phase)
   int counter_;
   Mode mode_pr_;        // current control mode (PR or AB for ankle control)
   uint8_t mode_machine_;// current mode of robot (from robot state)

   // Gamepad (remote controller) for optional user input
//    Gamepad gamepad_;
//    REMOTE_DATA_RX rx_;   // raw data from remote

   // Data buffers for exchanging data between threads
   DataBuffer<MotorState> motor_state_buffer_;
   DataBuffer<MotorCommand> motor_command_buffer_;
   DataBuffer<ImuState> imu_state_buffer_;

   // DDS Publisher and Subscribers for LowCmd, LowState, and IMU (torso)
   ChannelPublisherPtr<unitree_hg::msg::dds_::LowCmd_> lowcmd_publisher_;
   ChannelSubscriberPtr<unitree_hg::msg::dds_::LowState_> lowstate_subscriber_;
   ChannelSubscriberPtr<unitree_hg::msg::dds_::IMUState_> imutorso_subscriber_;

   // Threads for sending commands and control loop
   ThreadPtr command_writer_thread_;
   ThreadPtr control_thread_;

   // Motion switcher client to release any existing control mode on the robot
   std::shared_ptr<b2::MotionSwitcherClient> msc_;

 public:
   // Constructor: initialize DDS communication and start control threads
   G1Example(const std::string &networkInterface)
       : time_(0.0),
         control_dt_(0.002),
         duration_(3.0),
         counter_(0),
         mode_pr_(Mode::PR),
         mode_machine_(0) 
   {
       // Initialize DDS communication on the given network interface (domain 0)
       ChannelFactory::Instance()->Init(0, networkInterface);

       // Try to shut down any existing motion control on the robot for safety
       msc_ = std::make_shared<b2::MotionSwitcherClient>();
       msc_->SetTimeout(5.0f);
       msc_->Init();
       std::string form, name;
       // Loop until the robot's built-in controller is released
       while (msc_->CheckMode(form, name), !name.empty()) {
           int ret = msc_->ReleaseMode();
           if (ret != 0) {
               std::cout << "Failed to release mode. Retrying in 5 seconds..." << std::endl;
           } else {
               std::cout << "Released motion control mode." << std::endl;
           }
           sleep(5);
       }

        // Create DDS publisher for low-level commands
        lowcmd_publisher_.reset(new ChannelPublisher<unitree_hg::msg::dds_::LowCmd_>("rt/lowcmd"));
        lowcmd_publisher_->InitChannel();

        // Create DDS subscriber for low-level state messages
        lowstate_subscriber_.reset(new ChannelSubscriber<unitree_hg::msg::dds_::LowState_>("rt/lowstate"));
        lowstate_subscriber_->InitChannel(std::bind(&G1Example::LowStateHandler, this, std::placeholders::_1), 1);

        // Create DDS subscriber for torso IMU messages (if needed)
        imutorso_subscriber_.reset(new ChannelSubscriber<unitree_hg::msg::dds_::IMUState_>("rt/secondary_imu"));
        imutorso_subscriber_->InitChannel(std::bind(&G1Example::imuTorsoHandler, this, std::placeholders::_1), 1);

        // Launch recurrent threads for command writing and control loop at 2ms interval
        command_writer_thread_ = CreateRecurrentThreadEx("command_writer", UT_CPU_ID_NONE, 2000, &G1Example::LowCommandWriter, this);
        control_thread_        = CreateRecurrentThreadEx("control",         UT_CPU_ID_NONE, 2000, &G1Example::Control,         this);
        // Note: These threads will run in parallel with the simulation's Control_loop.
   }

   // Callback for IMU torso data (prints torso orientation periodically for debugging)
   void imuTorsoHandler(const void *message) {
       const unitree_hg::msg::dds_::IMUState_ &imu_torso = *(const unitree_hg::msg::dds_::IMUState_*)message;
       auto &rpy = imu_torso.rpy();
       if (counter_ % 500 == 0) {
           printf("IMU.torso.rpy: %.2f %.2f %.2f\n", rpy[0], rpy[1], rpy[2]);
       }
       // No state buffering needed for torso IMU in this example
   }

   // Callback for LowState messages from the robot (updates motor and IMU state buffers)
   void LowStateHandler(const void *message) {
       const unitree_hg::msg::dds_::LowState_ &low_state = *(const unitree_hg::msg::dds_::LowState_*)message;
       // Verify data integrity using CRC
       if (low_state.crc() != Crc32Core((uint32_t *)&low_state, (sizeof(low_state) >> 2) - 1)) {
           std::cout << "[ERROR] CRC Error in received LowState" << std::endl;
           return;
       }
       // Retrieve motor states from LowState and store in buffer
       MotorState ms_tmp;
       for (int i = 0; i < G1_NUM_MOTOR; ++i) {
           ms_tmp.q.at(i)  = low_state.motor_state()[i].q();
           ms_tmp.dq.at(i) = low_state.motor_state()[i].dq();
           // (Optional: check motor health code)
           if (low_state.motor_state()[i].motorstate() && i <= RightAnkleRoll) {
               std::cout << "[ERROR] Motor " << i << " error code: " 
                         << low_state.motor_state()[i].motorstate() << std::endl;
           }
       }
       motor_state_buffer_.SetData(ms_tmp);
       // Retrieve IMU (pelvis) state
       ImuState imu_tmp;
       imu_tmp.omega = low_state.imu_state().gyroscope();
       imu_tmp.rpy   = low_state.imu_state().rpy();
       imu_state_buffer_.SetData(imu_tmp);
       // Update gamepad (wireless remote) state
    //    memcpy(rx_.buff, &low_state.wireless_remote()[0], sizeof(rx_.buff));
    //    gamepad_.update(rx_.RF_RX);
       // Update robot type/mode if changed
       if (mode_machine_ != low_state.mode_machine()) {
           if (mode_machine_ == 0) {
               std::cout << "G1 mode_machine changed to: " << unsigned(low_state.mode_machine()) << std::endl;
           }
           mode_machine_ = low_state.mode_machine();
       }
       // Periodically print some status (every 1 second)
       if (++counter_ % 500 == 0) {
           counter_ = 0;
           auto &rpy = low_state.imu_state().rpy();
           printf("IMU.pelvis.rpy: %.2f %.2f %.2f\n", rpy[0], rpy[1], rpy[2]);
        //    printf("Gamepad buttons: A=%d, B=%d, X=%d, Y=%d\n",
        //           (int)gamepad_.A.pressed, (int)gamepad_.B.pressed, 
        //           (int)gamepad_.X.pressed, (int)gamepad_.Y.pressed);
       }
   }

   // Thread function: writes motor commands to the robot at each control cycle
   void LowCommandWriter() {
       if (!s_isRunning) {
           return;  // Only publish commands while simulation/program is running
       }
       unitree_hg::msg::dds_::LowCmd_ dds_low_command;
       // Set the mode bits in the LowCmd message
       dds_low_command.mode_pr() = static_cast<uint8_t>(mode_pr_);
       dds_low_command.mode_machine() = mode_machine_;
       // Fetch the latest motor command from the control loop
       std::shared_ptr<const MotorCommand> mc = motor_command_buffer_.GetData();
       if (mc) {
           // Populate the LowCmd message with desired motor commands
           for (size_t i = 0; i < G1_NUM_MOTOR; i++) {
               dds_low_command.motor_cmd()[i].mode() = 1;  // 1: Enable motor, 0: Disable
               dds_low_command.motor_cmd()[i].tau()  = mc->tau_ff.at(i);
               dds_low_command.motor_cmd()[i].q()    = mc->q_target.at(i);
               dds_low_command.motor_cmd()[i].dq()   = mc->dq_target.at(i);
               dds_low_command.motor_cmd()[i].kp()   = mc->kp.at(i);
               dds_low_command.motor_cmd()[i].kd()   = mc->kd.at(i);
           }
           // Compute CRC for LowCmd and write to DDS
           dds_low_command.crc() = Crc32Core((uint32_t *)&dds_low_command, (sizeof(dds_low_command) >> 2) - 1);
           lowcmd_publisher_->Write(dds_low_command);
       }
   }

   // Thread function: control loop that computes motor commands (runs at 2ms interval)
   void Control() {
       if (!wakeup) {
           return;  // Do nothing if simulation/program has stopped
       }
       MotorCommand motor_command_tmp;
       // Get latest motor state (positions) from buffer
       std::shared_ptr<const MotorState> ms = motor_state_buffer_.GetData();
       // Initialize command with default values (zero positions, set stiffness/damping)
       for (int i = 0; i < G1_NUM_MOTOR; ++i) {
           motor_command_tmp.tau_ff[i]   = 0.0f;
           motor_command_tmp.q_target[i] = 0.0f;
           motor_command_tmp.dq_target[i] = 0.0f;
           motor_command_tmp.kp[i] = Kp[i];
           motor_command_tmp.kd[i] = Kd[i];
       }
       if (ms) {
           // Increment time by control step
           time_ += control_dt_;
           if (time_ < duration_) {
               // [Stage 1]: smoothly move robot to zero posture over `duration_` seconds
               double ratio = std::clamp(time_ / duration_, 0.0, 1.0);
               for (int i = 0; i < G1_NUM_MOTOR; ++i) {
                   // Interpolate from current position to 0 based on ratio
                   motor_command_tmp.q_target[i] = (1.0 - ratio) * ms->q[i];
               }
           } else {
               // *** New motion capture control ***
               mode_pr_ = Mode::PR;
               double t_since = time_ - duration_;
               double ramp_factor = std::clamp(t_since / 1.0, 0.0, 1.0);
               float ramp = static_cast<float>(ramp_factor);
               float scale = 0.5;
               // Update only the specified joints (shoulders and elbows)
               motor_command_tmp.q_target[LeftShoulderPitch] = (float) G1_hardware_signal.left_shoulder_pitch * scale;
               motor_command_tmp.q_target[LeftShoulderRoll]  = (float) G1_hardware_signal.left_shoulder_roll * scale;
               motor_command_tmp.q_target[LeftShoulderYaw]   = (float) G1_hardware_signal.left_shoulder_yaw * scale;
               motor_command_tmp.q_target[LeftElbow]         = (float) G1_hardware_signal.left_elbow_yaw * scale;
               motor_command_tmp.q_target[RightShoulderPitch] = (float) G1_hardware_signal.right_shoulder_pitch * scale;
               motor_command_tmp.q_target[RightShoulderRoll]  = (float) G1_hardware_signal.right_shoulder_roll * scale;
               motor_command_tmp.q_target[RightShoulderYaw]   = (float) G1_hardware_signal.right_shoulder_yaw * scale;
               motor_command_tmp.q_target[RightElbow]         = (float) G1_hardware_signal.right_elbow_yaw * scale;
               // Dynamically adjust stiffness (kp) and damping (kd) for smooth control
               motor_command_tmp.kp[LeftShoulderPitch]   = Kp[LeftShoulderPitch] * ramp;
               motor_command_tmp.kd[LeftShoulderPitch]   = Kd[LeftShoulderPitch] * ramp;
               motor_command_tmp.kp[LeftShoulderRoll]    = Kp[LeftShoulderRoll] * ramp;
               motor_command_tmp.kd[LeftShoulderRoll]    = Kd[LeftShoulderRoll] * ramp;
               motor_command_tmp.kp[LeftShoulderYaw]     = Kp[LeftShoulderYaw] * ramp;
               motor_command_tmp.kd[LeftShoulderYaw]     = Kd[LeftShoulderYaw] * ramp;
               motor_command_tmp.kp[LeftElbow]           = Kp[LeftElbow] * ramp;
               motor_command_tmp.kd[LeftElbow]           = Kd[LeftElbow] * ramp;
               motor_command_tmp.kp[RightShoulderPitch]  = Kp[RightShoulderPitch] * ramp;
               motor_command_tmp.kd[RightShoulderPitch]  = Kd[RightShoulderPitch] * ramp;
               motor_command_tmp.kp[RightShoulderRoll]   = Kp[RightShoulderRoll] * ramp;
               motor_command_tmp.kd[RightShoulderRoll]   = Kd[RightShoulderRoll] * ramp;
               motor_command_tmp.kp[RightShoulderYaw]    = Kp[RightShoulderYaw] * ramp;
               motor_command_tmp.kd[RightShoulderYaw]    = Kd[RightShoulderYaw] * ramp;
               motor_command_tmp.kp[RightElbow]          = Kp[RightElbow] * ramp;
               motor_command_tmp.kd[RightElbow]          = Kd[RightElbow] * ramp;
           }
           // Update the command buffer with new targets (to be sent by LowCommandWriter)
           motor_command_buffer_.SetData(motor_command_tmp);
       }
   }
};




/*************************************************Kinect Render, display human skeleton joint tracking*********************************************/

Visualization::Layout3d s_layoutMode = Visualization::Layout3d::OnlyMainView;
bool s_visualizeJointFrame = false;
k4abt_frame_t globalBodyFrameForSkeleton = nullptr;

void PrintUsage()
{
    printf("\n");
    printf(" Basic Navigation:\n\n");
    printf(" Rotate: Rotate the camera by moving the mouse while holding mouse left button\n");
    printf(" Pan: Translate the scene by holding Ctrl key and drag the scene with mouse left button\n");
    printf(" Zoom in/out: Move closer/farther away from the scene center by scrolling the mouse scroll wheel\n");
    printf(" Select Center: Center the scene based on a detected joint by right clicking the joint with mouse\n");
    printf("\n");
    printf(" Key Shortcuts\n\n");
    printf(" ESC: quit\n");
    printf(" h: help\n");
    printf(" b: body visualization mode\n");
    printf(" k: 3d window layout\n");
    printf("\n");
}

int64_t ProcessKey(void* /*context*/, int key)
{
    // https://www.glfw.org/docs/latest/group__keys.html
    switch (key)
    {
        // Quit
    case GLFW_KEY_ESCAPE:
        s_isRunning = false;
        break;
    case GLFW_KEY_K:
        s_layoutMode = (Visualization::Layout3d)(((int)s_layoutMode + 1) % (int)Visualization::Layout3d::Count);
        break;
    case GLFW_KEY_B:
        s_visualizeJointFrame = !s_visualizeJointFrame;
        break;
    case GLFW_KEY_H:
        PrintUsage();
        break;
    }
    return 1;
}

int64_t CloseCallback(void* /*context*/)
{
    s_isRunning = false;
    return 1;
}

void renderSkeletonAndPointCloud(k4abt_frame_t bodyFrame, Window3dWrapper &kinectRenderWindow, int depthWidth, int depthHeight, int step = 2) {
    uint32_t numBodies = k4abt_frame_get_num_bodies(bodyFrame);
    float minDistance = std::numeric_limits<float>::max();
    uint32_t closestBodyIndex = 0;

    if (numBodies == 0) {
        return;
    }
    
    for (uint32_t i = 0; i < numBodies; i++) {
        k4abt_body_t body;
        if (k4abt_frame_get_body_skeleton(bodyFrame, i, &body.skeleton) != K4A_RESULT_SUCCEEDED) {
            std::cerr << "Get skeleton from body frame failed!" << std::endl;
            continue;
        }

        k4abt_joint_t spineChestJoint = body.skeleton.joints[K4ABT_JOINT_SPINE_CHEST];
        float distance = spineChestJoint.position.xyz.z;

        if (distance < minDistance) {
            minDistance = distance;
            closestBodyIndex = i;
        }
    }

    // Obtain original capture that generates the body tracking result
    k4a_capture_t originalCapture = k4abt_frame_get_capture(bodyFrame);
    k4a_image_t depthImage = k4a_capture_get_depth_image(originalCapture);
    std::vector<Color> pointCloudColors(depthWidth * depthHeight, { 1.f, 1.f, 1.f, 1.f });
    // Read body index map and assign colors
    k4a_image_t bodyIndexMap = k4abt_frame_get_body_index_map(bodyFrame);
    const uint8_t* bodyIndexMapBuffer = k4a_image_get_buffer(bodyIndexMap);
    for (int i = 0; i < depthWidth * depthHeight; i = i+step) {
        uint8_t bodyIndex = bodyIndexMapBuffer[i];
        if (bodyIndex == closestBodyIndex) {
            uint32_t bodyId = k4abt_frame_get_body_id(bodyFrame, bodyIndex);
            pointCloudColors[i] = g_bodyColors[bodyId % g_bodyColors.size()];
        }
    }
    k4a_image_release(bodyIndexMap);
    kinectRenderWindow.UpdatePointClouds(depthImage, pointCloudColors);//add parameter 'int step' to accelerate point cloud render
    k4a_capture_release(originalCapture);
    k4a_image_release(depthImage);


    kinectRenderWindow.CleanJointsAndBones();
    k4abt_body_t body;
    VERIFY(k4abt_frame_get_body_skeleton(bodyFrame, closestBodyIndex, &body.skeleton), "Get skeleton from body frame failed!");

    // Assign the correct color based on the body id
    Color color = g_bodyColors[ 1 % g_bodyColors.size()];
    color.a = 0.4f;
    Color lowConfidenceColor = g_bodyColors[ 6 % g_bodyColors.size()];
    lowConfidenceColor.a = 0.3f;

    // Visualize joints
    for (int joint = 0; joint < static_cast<int>(K4ABT_JOINT_COUNT); joint++) {
        if (body.skeleton.joints[joint].confidence_level >= K4ABT_JOINT_CONFIDENCE_MEDIUM) {
            const k4a_float3_t& jointPosition = body.skeleton.joints[joint].position;
            const k4a_quaternion_t& jointOrientation = body.skeleton.joints[joint].orientation;

            kinectRenderWindow.AddJoint(
                jointPosition,
                jointOrientation,
                body.skeleton.joints[joint].confidence_level >= K4ABT_JOINT_CONFIDENCE_MEDIUM ? color : lowConfidenceColor);
        }
    }

    // Visualize bones
    for (size_t boneIdx = 0; boneIdx < g_boneList.size(); boneIdx++) {
        k4abt_joint_id_t joint1 = g_boneList[boneIdx].first;
        k4abt_joint_id_t joint2 = g_boneList[boneIdx].second;

        if (body.skeleton.joints[joint1].confidence_level >= K4ABT_JOINT_CONFIDENCE_LOW &&
            body.skeleton.joints[joint2].confidence_level >= K4ABT_JOINT_CONFIDENCE_LOW) {
            bool confidentBone = body.skeleton.joints[joint1].confidence_level >= K4ABT_JOINT_CONFIDENCE_MEDIUM &&
                                 body.skeleton.joints[joint2].confidence_level >= K4ABT_JOINT_CONFIDENCE_MEDIUM;
            const k4a_float3_t& joint1Position = body.skeleton.joints[joint1].position;
            const k4a_float3_t& joint2Position = body.skeleton.joints[joint2].position;

            kinectRenderWindow.AddBone(joint1Position, joint2Position, confidentBone ? color : lowConfidenceColor);
        }
    }
}

void KinectRender_loop(k4a_calibration_t sensorCalibration) {
    std::cout<<"Kinect Render loop start..."<<std::endl;
    std::cout<<"Please use the wake-up action to start or stop the TeleOperation..."<<std::endl;
    Window3dWrapper kinectRenderWindow;
    kinectRenderWindow.Create("Kinect Render", sensorCalibration);
    kinectRenderWindow.SetCloseCallback(CloseCallback);
    kinectRenderWindow.SetKeyCallback(ProcessKey);
    int depthWidth = sensorCalibration.depth_camera_calibration.resolution_width;
    int depthHeight = sensorCalibration.depth_camera_calibration.resolution_height;

    time_point<high_resolution_clock> KinectRender_start;
    while (s_isRunning) {
        if (globalBodyFrameForSkeleton != nullptr) {
            renderSkeletonAndPointCloud(globalBodyFrameForSkeleton, kinectRenderWindow, depthWidth, depthHeight);
            k4abt_frame_release(globalBodyFrameForSkeleton);
            globalBodyFrameForSkeleton = nullptr;

            kinectRenderWindow.SetLayout3d(s_layoutMode);
            kinectRenderWindow.SetJointFrameVisualization(s_visualizeJointFrame);
            kinectRenderWindow.Render();

            #if EchoFrequency
            time_point<high_resolution_clock> KinectRender_end = high_resolution_clock::now();
            auto duration = duration_cast<microseconds>(KinectRender_end - KinectRender_start).count();
            KinectRender_start = KinectRender_end;
            double frequency = 1e6 / duration;
            std::cout << "Kinect Render loop: " << frequency << " Hz" << std::endl;
            #endif
        } 
        else {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }
    kinectRenderWindow.Delete();
}

/*************************************************Simulation Unitree Robot Control************************************************/

mjModel* m = nullptr;               // MuJoCo model
mjData*  d = nullptr;               // MuJoCo data

GLFWwindow* mujocoRenderWindow;
mjvCamera cam;                      // abstract camera
mjvOption opt;                      // visualization options
mjvScene scn;                       // abstract scene
mjrContext con;                     // custom GPU context

// mouse interaction
bool button_left = false;
bool button_middle = false;
bool button_right =  false;
double lastx = 0;
double lasty = 0;

// mouse button callback
void mouse_button(GLFWwindow* window, int button, int act, int mods) {
  button_left = (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT)==GLFW_PRESS);
  button_middle = (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_MIDDLE)==GLFW_PRESS);
  button_right = (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT)==GLFW_PRESS);
  glfwGetCursorPos(window, &lastx, &lasty);
}
// mouse move callback
void mouse_move(GLFWwindow* window, double xpos, double ypos) {
  // no buttons down: nothing to do
  if (!button_left && !button_middle && !button_right) {
    return;
  }

  double dx = xpos - lastx;
  double dy = ypos - lasty;
  lastx = xpos;
  lasty = ypos;

  int width, height;
  glfwGetWindowSize(window, &width, &height);

  bool mod_shift = (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT)==GLFW_PRESS ||
                    glfwGetKey(window, GLFW_KEY_RIGHT_SHIFT)==GLFW_PRESS);

  mjtMouse action;
  if (button_right) {
    action = mod_shift ? mjMOUSE_MOVE_H : mjMOUSE_MOVE_V;
  } else if (button_left) {
    action = mod_shift ? mjMOUSE_ROTATE_H : mjMOUSE_ROTATE_V;
  } else {
    action = mjMOUSE_ZOOM;
  }

  mjv_moveCamera(m, action, dx/height, dy/height, &scn, &cam);
}
// scroll callback
void scroll(GLFWwindow* window, double xoffset, double yoffset) {
  // emulate vertical mouse motion = 5% of window height
  mjv_moveCamera(m, mjMOUSE_ZOOM, 0, -0.05*yoffset, &scn, &cam);
}

GLFWwindow * InitWindow() {
    if (!glfwInit())
        mju_error("can not initialize GLFW");
    GLFWwindow* window = glfwCreateWindow(1280, 960, "Mujoco Render", NULL, NULL);
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    // initialize visualization data structures
    mjv_defaultCamera(&cam);
    mjv_defaultOption(&opt);
    mjv_defaultScene(&scn);
    mjr_defaultContext(&con);

    // create scene and context
    mjv_makeScene(m, &scn, 2000);
    mjr_makeContext(m, &con, mjFONTSCALE_150);

    // install GLFW mouse and keyboard callbacks
    glfwSetCursorPosCallback(window, mouse_move);
    glfwSetMouseButtonCallback(window, mouse_button);
    glfwSetScrollCallback(window, scroll);
    return window;
}
void initCamera(mjvCamera* camera) {
    camera->lookat[0] = 0.0; 
    camera->lookat[1] = 0.0; 
    camera->lookat[2] = 0.5; 

    camera->distance = 3.0;

    camera->azimuth = 0;  
    camera->elevation = -30;
}

void DrawOneFrame(GLFWwindow* window) {
    // get framebuffer viewport
    glfwMakeContextCurrent(window);
    mjrRect viewport = {0, 0, 0, 0};
    glfwGetFramebufferSize(window, &viewport.width, &viewport.height);

    // update scene and render
    mjv_updateScene(m, d, &opt, NULL, &cam, mjCAT_ALL, &scn);
    mjr_render(viewport, &scn, &con);

    glfwSwapBuffers(window);
    glfwPollEvents();
}

void MujocoRender_loop() {
    std::cout<<"Mujoco Render loop start..."<<std::endl;
    mujocoRenderWindow = InitWindow();
    initCamera(&cam);
    DrawOneFrame(mujocoRenderWindow);

    time_point<high_resolution_clock> mujocoRender_start;
    while (s_isRunning) {
        DrawOneFrame(mujocoRenderWindow);

        #if EchoFrequency
        time_point<high_resolution_clock> mujocoRender_end = high_resolution_clock::now();
        auto duration = duration_cast<microseconds>(mujocoRender_end - mujocoRender_start).count();
        double frequency = 1e6 / duration;
        mujocoRender_start = mujocoRender_end;
        std::cout << "Mujoco Render loop: " << frequency << " Hz" << std::endl;
        #endif
    }

    mjv_freeScene(&scn);
    mjr_freeContext(&con);
    glfwDestroyWindow(mujocoRenderWindow);
    mj_deleteData(d);
    mj_deleteModel(m);
}

/*****************************************************Real Unitree Robot Control*****************************************************************/
#if Real_Control

#if Control_G1
// Replace there with the Unitree SDK control code
// Send your motor joint value by G1_hardware_signal
// Please refer to https://github.com/unitreerobotics/unitree_sdk2
#endif

#if Control_H1
// Replace there with the Unitree SDK control code
// Send your motor joint value by H1_hardware_signal
// Please refer to https://github.com/unitreerobotics/unitree_sdk2
#endif

#endif

/*****************************************************Control Smooth and Transfer*****************************************************************/

void Control_loop() {
    std::cout<<"control loop start..."<<std::endl;
    int left_shoulder_roll_joint_id = mj_name2id(m, mjOBJ_ACTUATOR, "left_shoulder_roll_joint");
    int left_shoulder_pitch_joint_id = mj_name2id(m, mjOBJ_ACTUATOR, "left_shoulder_pitch_joint");
    int left_shoulder_yaw_joint_id = mj_name2id(m, mjOBJ_ACTUATOR, "left_shoulder_yaw_joint");
    int right_shoulder_roll_joint_id = mj_name2id(m, mjOBJ_ACTUATOR, "right_shoulder_roll_joint");
    int right_shoulder_pitch_joint_id = mj_name2id(m, mjOBJ_ACTUATOR, "right_shoulder_pitch_joint");
    int right_shoulder_yaw_joint_id = mj_name2id(m, mjOBJ_ACTUATOR, "right_shoulder_yaw_joint");
    #if Control_H1
    int left_elbow_pitch_joint_id = mj_name2id(m, mjOBJ_ACTUATOR, "left_elbow_joint");
    int right_elbow_pitch_joint_id = mj_name2id(m, mjOBJ_ACTUATOR, "right_elbow_joint");
    #elif Control_G1
    int left_elbow_pitch_joint_id = mj_name2id(m, mjOBJ_ACTUATOR, "left_elbow_pitch_joint");
    int right_elbow_pitch_joint_id = mj_name2id(m, mjOBJ_ACTUATOR, "right_elbow_pitch_joint");
    #endif
    #if Enable_Torso
    int torso_joint_id = mj_name2id(m, mjOBJ_ACTUATOR, "torso_joint");
    #endif
    
    mj_step(m, d); // For starting render mujoco

    MovingAverageFilter ls_r_filter,ls_p_filter,ls_y_filter,
                        rs_r_filter,rs_p_filter,rs_y_filter,
                        le_y_filter,re_y_filter,sc_p_filter;

    StartEndPoseDetector pose_detector;

    time_point<high_resolution_clock> ctrl_start;
    while (s_isRunning) {
        double left_shoulder_roll = LS_mappingCameraRoll2RobotRadians(ls_r);
        double left_shoulder_pitch = LS_mappingCameraPitch2RobotRadians(ls_p);
        double left_shoulder_yaw = LS_mappingCameraYaw2RobotRadians(ls_y);
        double right_shoulder_roll = RS_mappingCameraRoll2RobotRadians(rs_r);
        double right_shoulder_pitch = RS_mappingCameraPitch2RobotRadians(rs_p);
        double right_shoulder_yaw = RS_mappingCameraYaw2RobotRadians(rs_y);
        double left_elbow_yaw = LE_mappingCameraYaw2RobotRadians(le_y);
        double right_elbow_yaw = RE_mappingCameraYaw2RobotRadians(re_y);

        #if Enable_Torso
        double spine_chest_torso = SC_mappingCameraPitch2RobotTorso(sc_p);
        #endif

        wakeup = pose_detector.isStartEndPose(left_shoulder_roll, left_shoulder_pitch, left_shoulder_yaw,
                                                  right_shoulder_roll, right_shoulder_pitch, right_shoulder_yaw,
                                                  left_elbow_yaw, right_elbow_yaw);

        if(!wakeup)
        {
        wakeup = pose_detector.isStartEndPose(left_shoulder_roll, left_shoulder_pitch, left_shoulder_yaw,
                                                  right_shoulder_roll, right_shoulder_pitch, right_shoulder_yaw,
                                                  left_elbow_yaw, right_elbow_yaw);
        }
        else
        {
            // smoothing
            left_shoulder_roll = ls_r_filter.update(left_shoulder_roll); // this is left_shoulder_roll, but didn't get send to the robot
            left_shoulder_pitch = ls_p_filter.update(left_shoulder_pitch) * 0; // this seems to be left_shoulder_roll
            left_shoulder_yaw = ls_y_filter.update(left_shoulder_yaw) * 0; // this left_shoulder_yaw, work correctly
            right_shoulder_roll = rs_r_filter.update(right_shoulder_roll) * 0;
            right_shoulder_pitch = rs_p_filter.update(right_shoulder_pitch) * 0;
            right_shoulder_yaw = rs_y_filter.update(right_shoulder_yaw) * 0;
            left_elbow_yaw = le_y_filter.update(left_elbow_yaw) * 0; // this is left_elbow_yaw, correct
            right_elbow_yaw = re_y_filter.update(right_elbow_yaw) * 0;
            #if Enable_Torso
            spine_chest_torso = sc_p_filter.update(spine_chest_torso);
            #endif

            mj_step1(m, d);
            // The coordinate system of the robot joints is as follows: the x-axis (roll) points forward, the y-axis (pitch) points left, 
            // and the z-axis (yaw) points upward. 
            // Test to get the rotation order, because the coordinate system definition of the Kinect camera is still a mystery
            // Camera:   z(yaw)      x(roll)   y(pitch) 
            //             |           |          |
            //             ∨           ∨          ∨
            // Robot :  y(pitch)     z(yaw)    x(roll)
            d->ctrl[left_shoulder_pitch_joint_id] = left_shoulder_yaw;
            d->ctrl[left_shoulder_roll_joint_id] = left_shoulder_pitch;
            d->ctrl[left_shoulder_yaw_joint_id] = left_shoulder_roll;
            d->ctrl[right_shoulder_pitch_joint_id] = right_shoulder_yaw;
            d->ctrl[right_shoulder_roll_joint_id] = right_shoulder_pitch;
            d->ctrl[right_shoulder_yaw_joint_id] = right_shoulder_roll;
            d->ctrl[left_elbow_pitch_joint_id] = left_elbow_yaw;
            d->ctrl[right_elbow_pitch_joint_id] = right_elbow_yaw;
            #if Enable_Torso
            d->ctrl[torso_joint_id] = spine_chest_torso;
            #endif
            mj_step2(m, d);

            #if Real_Control
            #if Control_H1
            H1_hardware_signal.left_shoulder_pitch = left_shoulder_yaw;
            H1_hardware_signal.left_shoulder_roll = left_shoulder_roll;
            H1_hardware_signal.left_shoulder_yaw = left_shoulder_pitch;
            H1_hardware_signal.right_shoulder_pitch = right_shoulder_yaw;
            H1_hardware_signal.right_shoulder_roll = right_shoulder_pitch;
            H1_hardware_signal.right_shoulder_yaw = right_shoulder_roll;
            H1_hardware_signal.left_elbow_yaw = left_elbow_yaw;
            H1_hardware_signal.right_elbow_yaw = right_elbow_yaw;
            #elif Control_G1
            G1_hardware_signal.left_shoulder_pitch = left_shoulder_yaw;
            G1_hardware_signal.left_shoulder_roll = left_shoulder_roll;
            G1_hardware_signal.left_shoulder_yaw = left_shoulder_pitch;
            G1_hardware_signal.right_shoulder_pitch = right_shoulder_yaw;
            G1_hardware_signal.right_shoulder_roll = right_shoulder_pitch;
            G1_hardware_signal.right_shoulder_yaw = right_shoulder_roll;
            G1_hardware_signal.left_elbow_yaw = left_elbow_yaw;
            G1_hardware_signal.right_elbow_yaw = right_elbow_yaw;
            #endif
            #endif

            #if EchoFrequency
            time_point<high_resolution_clock> ctrl_end = high_resolution_clock::now();
            auto duration = duration_cast<microseconds>(ctrl_end - ctrl_start).count();
            double frequency = 1e6 / duration;
            ctrl_start = ctrl_end;
            std::cout << "Control loop: " << frequency << " Hz" << std::endl;
            #endif
        }
    }
}


// Process the newly captured skeleton point data to calculate the joint angles that control the robot
void ProcessNewSkeletonData(k4abt_frame_t bodyFrame) {
    uint32_t numBodies = k4abt_frame_get_num_bodies(bodyFrame);
    float minDistance = std::numeric_limits<float>::max();
    uint32_t closestBodyIndex = 0;

    if (numBodies == 0) {
        return;
    }
    // Take the data of the human body closest to the kinect camera as input
    for (uint32_t i = 0; i < numBodies; i++) {
        k4abt_body_t body;
        if (k4abt_frame_get_body_skeleton(bodyFrame, i, &body.skeleton) != K4A_RESULT_SUCCEEDED) {
            std::cerr << "Get skeleton from body frame failed!" << std::endl;
            continue;
        }

        k4abt_joint_t spineChestJoint = body.skeleton.joints[K4ABT_JOINT_SPINE_CHEST];
        float distance = spineChestJoint.position.xyz.z;

        if (distance < minDistance) {
            minDistance = distance;
            closestBodyIndex = i;
        }
    }

    k4abt_body_t closestBody;
    if (k4abt_frame_get_body_skeleton(bodyFrame, closestBodyIndex, &closestBody.skeleton) != K4A_RESULT_SUCCEEDED) {
        std::cerr << "Get skeleton from body frame failed!" << std::endl;
        return;
    }

    // temp var
    k4a_quaternion_t relativeJointOrientation;
    static k4a_quaternion_t SpineChest_orientation, LeftShoulder_orientation, RightShoulder_orientation;

    if(closestBody.skeleton.joints[K4ABT_JOINT_SPINE_CHEST].confidence_level > K4ABT_JOINT_CONFIDENCE_LOW){
        SpineChest_orientation = closestBody.skeleton.joints[K4ABT_JOINT_SPINE_CHEST].orientation;
        quaternion2Euler(SpineChest_orientation, sc_r, sc_p, sc_y);
    }

    if(closestBody.skeleton.joints[K4ABT_JOINT_SHOULDER_LEFT].confidence_level > K4ABT_JOINT_CONFIDENCE_LOW){
        LeftShoulder_orientation = closestBody.skeleton.joints[K4ABT_JOINT_SHOULDER_LEFT].orientation;
        relativeJointOrientation = calculateRelativeQuaternion(LeftShoulder_orientation, SpineChest_orientation);
        quaternion2Euler(relativeJointOrientation, ls_r, ls_p, ls_y);
    }

    if(closestBody.skeleton.joints[K4ABT_JOINT_ELBOW_LEFT].confidence_level > K4ABT_JOINT_CONFIDENCE_LOW){
        relativeJointOrientation = calculateRelativeQuaternion(closestBody.skeleton.joints[K4ABT_JOINT_ELBOW_LEFT].orientation, LeftShoulder_orientation);
        quaternion2Euler(relativeJointOrientation, le_r, le_p, le_y);
    }

    if(closestBody.skeleton.joints[K4ABT_JOINT_SHOULDER_RIGHT].confidence_level > K4ABT_JOINT_CONFIDENCE_LOW){
        RightShoulder_orientation = closestBody.skeleton.joints[K4ABT_JOINT_SHOULDER_RIGHT].orientation;
        relativeJointOrientation = calculateRelativeQuaternion(RightShoulder_orientation, SpineChest_orientation);
        quaternion2Euler(relativeJointOrientation, rs_r, rs_p, rs_y);

    }
    
    if(closestBody.skeleton.joints[K4ABT_JOINT_ELBOW_RIGHT].confidence_level > K4ABT_JOINT_CONFIDENCE_LOW){
        relativeJointOrientation = calculateRelativeQuaternion(closestBody.skeleton.joints[K4ABT_JOINT_ELBOW_RIGHT].orientation, RightShoulder_orientation);
        quaternion2Euler(relativeJointOrientation, re_r, re_p, re_y);
    }
    
    // For hand open and closing status detect
    #if Enable_Hand
    if(closestBody.skeleton.joints[K4ABT_JOINT_HANDTIP_LEFT].confidence_level > K4ABT_JOINT_CONFIDENCE_LOW && 
        closestBody.skeleton.joints[K4ABT_JOINT_WRIST_LEFT].confidence_level > K4ABT_JOINT_CONFIDENCE_LOW){
        k4a_quaternion_t LeftHandTip_orientation = closestBody.skeleton.joints[K4ABT_JOINT_HANDTIP_LEFT].orientation;
        k4a_quaternion_t LeftWrist_orientation = closestBody.skeleton.joints[K4ABT_JOINT_WRIST_LEFT].orientation;
        lh_a = calculateRelativeAngle(LeftHandTip_orientation, LeftWrist_orientation);
        if(lh_a < 30.0)
            std::cout<<"left hand open."<<std::endl;
        else if(lh_a > 50.0)
            std::cout<<"left hand close."<<std::endl;
    }
    else
        std::cout<<"left hand unknow."<<std::endl;

    if(closestBody.skeleton.joints[K4ABT_JOINT_HANDTIP_RIGHT].confidence_level > K4ABT_JOINT_CONFIDENCE_LOW && 
        closestBody.skeleton.joints[K4ABT_JOINT_WRIST_RIGHT].confidence_level > K4ABT_JOINT_CONFIDENCE_LOW){
        k4a_quaternion_t RightHandTip_orientation = closestBody.skeleton.joints[K4ABT_JOINT_HANDTIP_RIGHT].orientation;
        k4a_quaternion_t RightWrist_orientation = closestBody.skeleton.joints[K4ABT_JOINT_WRIST_RIGHT].orientation;
        rh_a = calculateRelativeAngle(RightHandTip_orientation, RightWrist_orientation);
        if(rh_a < 30.0)
            std::cout<<"right hand open."<<std::endl;
        else if(rh_a > 50.0)
            std::cout<<"right hand close."<<std::endl;
    }
    else
        std::cout<<"right hand unknow."<<std::endl;
    #endif
}

void Main_loop(){

    PrintUsage();

    k4a_device_t device = nullptr;
    VERIFY(k4a_device_open(0, &device), "Open K4A Device failed!");

    // Start camera. Make sure depth camera is enabled.
    k4a_device_configuration_t deviceConfig = K4A_DEVICE_CONFIG_INIT_DISABLE_ALL;
    // K4A_DEPTH_MODE_NFOV_2X2BINNED, /**< Depth captured at 320x288. Passive IR is also captured at 320x288. */
    // K4A_DEPTH_MODE_NFOV_UNBINNED,  /**< Depth captured at 640x576. Passive IR is also captured at 640x576. */
    // K4A_DEPTH_MODE_WFOV_2X2BINNED, /**< Depth captured at 512x512. Passive IR is also captured at 512x512. */
    deviceConfig.depth_mode = K4A_DEPTH_MODE_NFOV_UNBINNED;
    deviceConfig.color_resolution = K4A_COLOR_RESOLUTION_OFF;
    VERIFY(k4a_device_start_cameras(device, &deviceConfig), "Start K4A cameras failed!");

    // Get calibration information
    k4a_calibration_t sensorCalibration;
    VERIFY(k4a_device_get_calibration(device, deviceConfig.depth_mode, deviceConfig.color_resolution, &sensorCalibration),
        "Get depth camera calibration failed!");

    // Create Body Tracker
    k4abt_tracker_t tracker = nullptr;
    k4abt_tracker_configuration_t trackerConfig = K4ABT_TRACKER_CONFIG_DEFAULT;
    trackerConfig.processing_mode = K4ABT_TRACKER_PROCESSING_MODE_GPU_CUDA;
    VERIFY(k4abt_tracker_create(&sensorCalibration, trackerConfig, &tracker), "Body tracker initialization failed!");
    // Do not use Kinect's built-in smoothing
    k4abt_tracker_set_temporal_smoothing(tracker, 0.0);

    std::thread KinectRender_thread(KinectRender_loop, sensorCalibration);

    time_point<high_resolution_clock> main_start;
    while (s_isRunning)
    {
        k4a_capture_t sensorCapture = nullptr;
        k4a_wait_result_t getCaptureResult = k4a_device_get_capture(device, &sensorCapture, 0); // without blocking, timeout_in_ms is set to 0 
        if (getCaptureResult == K4A_WAIT_RESULT_SUCCEEDED)
        {
            k4a_wait_result_t queueCaptureResult = k4abt_tracker_enqueue_capture(tracker, sensorCapture, 0); // without blocking, timeout_in_ms is set to 0 
            k4a_capture_release(sensorCapture);
            if (queueCaptureResult == K4A_WAIT_RESULT_FAILED)
            {
                std::cout << "Error! Add capture to tracker process queue failed!" << std::endl;
                break;
            }
        }
        else if (getCaptureResult != K4A_WAIT_RESULT_TIMEOUT)
        {
            std::cout << "Get depth capture returned error: " << getCaptureResult << std::endl;
            break;
        }

        k4abt_frame_t bodyFrame = nullptr;
        k4a_wait_result_t popFrameResult = k4abt_tracker_pop_result(tracker, &bodyFrame, 0); // without blocking, timeout_in_ms is set to 0
        if (popFrameResult == K4A_WAIT_RESULT_SUCCEEDED)
        {
            if (globalBodyFrameForSkeleton == nullptr)
            {
                // Passing references for increased efficiency, manually managing reference counts
                globalBodyFrameForSkeleton = bodyFrame;
                k4abt_frame_reference(globalBodyFrameForSkeleton);
            }
            ProcessNewSkeletonData(bodyFrame);
            k4abt_frame_release(bodyFrame);

            #if EchoFrequency
            time_point<high_resolution_clock> main_end = high_resolution_clock::now();
            auto duration = duration_cast<milliseconds>(main_end - main_start).count();
            double frequency = 1e3 / duration;
            main_start = main_end;
            std::cout << "Main loop : " << frequency << " Hz" << std::endl;
            #endif
        }
    }

    std::cout << "kinect_teleoperate_robot finished!" << std::endl;

    k4abt_tracker_shutdown(tracker);
    k4abt_tracker_destroy(tracker);

    k4a_device_stop_cameras(device);
    k4a_device_close(device);

    KinectRender_thread.join();
}


int main(int argc, char** argv)
{

    if (argc < 2) {
        std::cout << "Usage: g1_ankle_swing_example network_interface" << std::endl;
        exit(0);
    }
    std::string networkInterface = argv[1];
    G1Example custom(networkInterface);
    
    #if Control_G1
    const char* model_path = "../src/unitree_g1/scene.xml";
    #endif
    #if Control_H1
    const char* model_path = "../src/unitree_h1/mjcf/scene.xml"; 
    #endif

    char error[1000];
    m = mj_loadXML(model_path, nullptr, error, 1000);
    d = mj_makeData(m);
    mj_resetData(m, d);

    std::thread control_thread(Control_loop);
    std::thread mujocoRender_thread(MujocoRender_loop);

    Main_loop();

    control_thread.join();
    mujocoRender_thread.join();

    return 0;
}
