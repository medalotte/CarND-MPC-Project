#include <math.h>
#include <uWS/uWS.h>
#include <chrono>
#include <iostream>
#include <string>
#include <thread>
#include <vector>
#include "Eigen-3.3/Eigen/Core"
#include "Eigen-3.3/Eigen/QR"
#include "json.hpp"
#include "MPC.h"

// For converting back and forth between radians and degrees.
constexpr double pi() { return M_PI; }
double deg2rad(double x) { return x * pi() / 180; }
double rad2deg(double x) { return x * 180 / pi(); }

// convert Eigen::VectorXd to std::vector<double>
std::vector<double> eigenv2stdv(const Eigen::VectorXd& v) {
    return std::vector<double>(v.data(), v.data() + v.size());
}

// Checks if the SocketIO event has JSON data.
// If there is data the JSON object in string format will be returned,
// else the empty string "" will be returned.
std::string hasData(std::string s) {
  auto found_null = s.find("null");
  auto b1 = s.find_first_of("[");
  auto b2 = s.rfind("}]");
  if (found_null != std::string::npos) {
    return "";
  } else if (b1 != std::string::npos && b2 != std::string::npos) {
    return s.substr(b1, b2 - b1 + 2);
  }
  return "";
}

// Evaluate a polynomial.
double polyeval(const Eigen::VectorXd &coeffs, double x) {
  double result = 0.0;
  for (int i = 0; i < coeffs.size(); ++i) {
    result += coeffs[i] * pow(x, i);
  }
  return result;
}

// Fit a polynomial.
// Adapted from:
// https://github.com/JuliaMath/Polynomials.jl/blob/master/src/Polynomials.jl#L676-L716
Eigen::VectorXd polyfit(const Eigen::VectorXd &xvals, const Eigen::VectorXd &yvals, int order) {
  assert(xvals.size() == yvals.size());
  assert(order >= 1 && order <= xvals.size() - 1);

  Eigen::MatrixXd A(xvals.size(), order + 1);

  for (int i = 0; i < xvals.size(); ++i) {
    A(i, 0) = 1.0;
  }

  for (int j = 0; j < xvals.size(); ++j) {
    for (int i = 0; i < order; ++i) {
      A(j, i + 1) = A(j, i) * xvals(j);
    }
  }

  auto Q = A.householderQr();
  auto result = Q.solve(yvals);

  return result;
}

int main() {
  uWS::Hub h;

  // MPC is initialized here!
  MPC mpc;

  h.onMessage([&mpc](uWS::WebSocket<uWS::SERVER> ws, char *data, size_t length,
                     uWS::OpCode opCode) {
    // "42" at the start of the message means there's a websocket message event.
    // The 4 signifies a websocket message
    // The 2 signifies a websocket event
    std::string sdata = std::string(data).substr(0, length);
    if (sdata.size() > 2 && sdata[0] == '4' && sdata[1] == '2') {
        std::string s = hasData(sdata);
      if (s != "") {
        auto j = nlohmann::json::parse(s);
        std::string event = j[0].get<std::string>();
        if (event == "telemetry") {
          // j[1] is the data JSON object
          std::vector<double> ptsx = j[1]["ptsx"];
          std::vector<double> ptsy = j[1]["ptsy"];
          double px = j[1]["x"];
          double py = j[1]["y"];
          double psi = j[1]["psi"];
          double v = j[1]["speed"];

          //--- dump input data
          std::cout << std::endl;
          std::cout << "--- reference path ---" << std::endl;
          for(size_t i = 0; i < ptsx.size(); i++) {
              std::cout << "(" << ptsx[i] << ", " << ptsy[i] << ")";
              if(i == ptsx.size() - 1) {
                  std::cout << std::endl;
              }
              else {
                  std::cout << ", " << std::endl;
              }
          }

          std::cout << "--- current state of vehicle ---" << std::endl;
          std::cout << "px : " << px << std::endl;
          std::cout << "py : " << py << std::endl;
          std::cout << "psi : " << psi << std::endl;
          std::cout << "v : " << v << std::endl;

          // calculate reference path as relative coordinate from vehicle
          const auto ref_size = ptsx.size();
          const auto cospsi   = std::cos(-psi);
          const auto sinpsi   = std::sin(-psi);
          const auto order    = 3;

          Eigen::VectorXd relative_ptsx(ref_size);
          Eigen::VectorXd relative_ptsy(ref_size);
          for(size_t i = 0; i < ref_size; i++) {
              const double dx = ptsx[i] - px;
              const double dy = ptsy[i] - py;
              relative_ptsx[i] = dx * cospsi - dy * sinpsi;
              relative_ptsy[i] = dy * cospsi + dx * sinpsi;
          }

          const auto coeffs = polyfit(relative_ptsx, relative_ptsy, order);

          /**
           * TODO: Calculate steering angle and throttle using MPC.
           * Both are in between [-1, 1].
           */
          double steer_value;
          double throttle_value;

          nlohmann::json msgJson;
          // NOTE: Remember to divide by deg2rad(25) before you send the 
          //   steering value back. Otherwise the values will be in between 
          //   [-deg2rad(25), deg2rad(25] instead of [-1, 1].
          msgJson["steering_angle"] = steer_value;
          msgJson["throttle"] = throttle_value;

          // Display the MPC predicted trajectory 
          std::vector<double> mpc_x_vals;
          std::vector<double> mpc_y_vals;

          /**
           * TODO: add (x,y) points to list here, points are in reference to 
           *   the vehicle's coordinate system the points in the simulator are 
           *   connected by a Green line
           */

          msgJson["mpc_x"] = mpc_x_vals;
          msgJson["mpc_y"] = mpc_y_vals;

          // calculate smoothed reference path
          const auto smoothing_ratio    = 3;
          const auto smoothing_ref_size = ref_size * smoothing_ratio;
          const auto smoothing_delta_x  = std::abs(relative_ptsx[ref_size - 1] - relative_ptsx[0]) / smoothing_ref_size;
          Eigen::VectorXd smoothed_relative_ptsx(smoothing_ref_size);
          Eigen::VectorXd smoothed_relative_ptsy(smoothing_ref_size);

          smoothed_relative_ptsx[0] = relative_ptsx[0];
          smoothed_relative_ptsy[0] = polyeval(coeffs, smoothed_relative_ptsx[0]);
          for(size_t i = 1; i < smoothing_ref_size; i++) {
              smoothed_relative_ptsx[i] = smoothed_relative_ptsx[i - 1] + smoothing_delta_x;
              smoothed_relative_ptsy[i] = polyeval(coeffs, smoothed_relative_ptsx[i]);
          }

          msgJson["next_x"] = eigenv2stdv(smoothed_relative_ptsx);
          msgJson["next_y"] = eigenv2stdv(smoothed_relative_ptsy);

          //--- dump output data
          std::cout << "--- output calc by MPC ---" << std::endl;
          auto msg = "42[\"steer\"," + msgJson.dump() + "]";
          std::cout << msg << std::endl;
          // Latency
          // The purpose is to mimic real driving conditions where
          //   the car does actuate the commands instantly.
          //
          // Feel free to play around with this value but should be to drive
          //   around the track with 100ms latency.
          //
          // NOTE: REMEMBER TO SET THIS TO 100 MILLISECONDS BEFORE SUBMITTING.
          std::this_thread::sleep_for(std::chrono::milliseconds(100));
          ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
        }  // end "telemetry" if
      } else {
        // Manual driving
          std::string msg = "42[\"manual\",{}]";
        ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
      }
    }  // end websocket if
  }); // end h.onMessage

  h.onConnection([&h](uWS::WebSocket<uWS::SERVER> ws, uWS::HttpRequest req) {
    std::cout << "Connected!!!" << std::endl;
  });

  h.onDisconnection([&h](uWS::WebSocket<uWS::SERVER> ws, int code,
                         char *message, size_t length) {
    ws.close();
    std::cout << "Disconnected" << std::endl;
  });

  int port = 4567;
  if (h.listen(port)) {
    std::cout << "Listening to port " << port << std::endl;
  } else {
    std::cerr << "Failed to listen to port" << std::endl;
    return -1;
  }
  
  h.run();
}
