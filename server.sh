export ISAACLAB_ROOT=$(python -c "import isaaclab, os; print(os.path.dirname(isaaclab.__file__))") 
export PYTHONPATH="$ISAACLAB_ROOT/source/isaaclab:$PYTHONPATH"
export PYTHONPATH="$ISAACLAB_ROOT/source/isaaclab:$ISAACLAB_ROOT/source/isaaclab_contrib:$PYTHONPATH"
export MATTERIX_PATH=/home/ubuntu/Matterix

# python scripts/matterix_server.py \
#     --task Matterix-Test-Beaker-Lift-Franka-v1 \
#     --host 0.0.0.0 \
#     --port 5555 \
#     --num-envs 1 \
#     --headless \
#     --position-scale 0.05 \
#     --rotation-scale 0.25 \
#     --primary-camera-name front_camera \
#     --wrist-camera-name wrist_camera \
#     --enable_cameras \
#     --lift-threshold 0.05

python scripts/matterix_server_beaker_cylinder.py \
    --task Matterix-Test-Beaker-Cylinder-Lift-Franka-v1 \
    --host 0.0.0.0 \
    --port 5555 \
    --num-envs 1 \
    --headless \
    --position-scale 0.05 \
    --rotation-scale 0.25 \
    --primary-camera-name front_camera \
    --wrist-camera-name wrist_camera \
    --enable_cameras \
    --lift-threshold 0.05