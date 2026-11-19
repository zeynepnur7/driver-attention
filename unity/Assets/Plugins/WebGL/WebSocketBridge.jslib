
mergeInto(LibraryManager.library, {
    ConnectWebSocket: function () {
        if (window.driverAttentionSocket &&
            window.driverAttentionSocket.readyState <= 1) {
            return;
        }

        var socket = new WebSocket("ws://127.0.0.1:8765");
        window.driverAttentionSocket = socket;

        socket.onopen = function () {
            console.log("Connected to Python WebSocket");
        };

        socket.onmessage = function (event) {
            SendMessage(
                "TelemetryManager",
                "ApplyTelemetry",
                event.data
            );
        };

        socket.onerror = function (error) {
            console.error("WebSocket error:", error);
        };

        socket.onclose = function () {
            console.log("WebSocket disconnected");
        };
    }
});
