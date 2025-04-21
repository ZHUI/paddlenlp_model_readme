
# MiniMax-Text-01
---


## README([From Huggingface](https://huggingface.co/MiniMaxAI/MiniMax-Text-01))

---
library_name: transformers
---
<div align="center">

<svg width="60%" height="auto" viewBox="0 0 144 48" fill="none" xmlns="http://www.w3.org/2000/svg">
<path d="M26.6782 7.96523C26.6782 7.02436 25.913 6.26087 24.9739 6.26087C24.0348 6.26087 23.2695 7.0261 23.2695 7.96523V36.2139C23.2695 38.4 21.4904 40.1791 19.3043 40.1791C17.1183 40.1791 15.3391 38.4 15.3391 36.2139V18.0904C15.3391 17.1496 14.5739 16.3861 13.6348 16.3861C12.6956 16.3861 11.9304 17.1513 11.9304 18.0904V25.7722C11.9304 27.9583 10.1513 29.7374 7.96518 29.7374C5.7791 29.7374 4 27.9583 4 25.7722V22.9878C4 22.3635 4.50609 21.8574 5.13043 21.8574C5.75478 21.8574 6.26087 22.3635 6.26087 22.9878V25.7722C6.26087 26.713 7.02605 27.4765 7.96518 27.4765C8.90431 27.4765 9.66954 26.7113 9.66954 25.7722V18.0904C9.66954 15.9044 11.4487 14.1252 13.6348 14.1252C15.8209 14.1252 17.6 15.9044 17.6 18.0904V36.2139C17.6 37.1548 18.3652 37.9183 19.3043 37.9183C20.2435 37.9183 21.0087 37.153 21.0087 36.2139V25.1322V7.96523C21.0087 5.77914 22.7878 4 24.9739 4C27.16 4 28.9391 5.77914 28.9391 7.96523V31.3565C28.9391 31.9809 28.433 32.487 27.8087 32.487C27.1843 32.487 26.6782 31.9809 26.6782 31.3565V7.96523ZM47.6539 14.1252C45.4678 14.1252 43.6887 15.9044 43.6887 18.0904V33.2296C43.6887 34.1704 42.9235 34.9339 41.9843 34.9339C41.0452 34.9339 40.28 34.1687 40.28 33.2296V7.96523C40.28 5.77914 38.5008 4 36.3148 4C34.1287 4 32.3496 5.77914 32.3496 7.96523V40.0348C32.3496 40.9756 31.5843 41.7391 30.6452 41.7391C29.7061 41.7391 28.9409 40.9739 28.9409 40.0348V36.0643C28.9409 35.44 28.4348 34.9339 27.8104 34.9339C27.1861 34.9339 26.68 35.44 26.68 36.0643V40.0348C26.68 42.2209 28.4591 44 30.6452 44C32.8313 44 34.6104 42.2209 34.6104 40.0348V7.96523C34.6104 7.02436 35.3756 6.26087 36.3148 6.26087C37.2539 6.26087 38.0191 7.0261 38.0191 7.96523V33.2296C38.0191 35.4156 39.7982 37.1948 41.9843 37.1948C44.1704 37.1948 45.9496 35.4156 45.9496 33.2296V18.0904C45.9496 17.1496 46.7148 16.3861 47.6539 16.3861C48.593 16.3861 49.3582 17.1513 49.3582 18.0904V31.3565C49.3582 31.9809 49.8643 32.487 50.4887 32.487C51.113 32.487 51.6191 31.9809 51.6191 31.3565V18.0904C51.6191 15.9044 49.84 14.1252 47.6539 14.1252Z" fill="url(#paint0_linear_17_483)"/>
<path d="M68.7671 16.5615H71.2541C71.3254 16.5615 71.3845 16.5859 71.435 16.6363C71.4836 16.6868 71.5097 16.7459 71.5097 16.8172V31.1824C71.5097 31.2537 71.4854 31.3128 71.435 31.3633C71.3845 31.4137 71.3254 31.4381 71.2541 31.4381H68.7671C68.6958 31.4381 68.6367 31.4137 68.5862 31.3633C68.5358 31.3146 68.5115 31.2537 68.5115 31.1824V21.812C68.5115 21.7563 68.4976 21.7268 68.4697 21.7268C68.4419 21.7268 68.4123 21.7476 68.3845 21.7911L66.1323 25.318C66.061 25.4311 65.9619 25.4885 65.8349 25.4885H64.581C64.4541 25.4885 64.3549 25.4328 64.2836 25.318L62.0315 21.7911C62.0036 21.7494 61.9741 21.7302 61.9462 21.7372C61.9184 21.7441 61.9045 21.7772 61.9045 21.8328V31.1824C61.9045 31.2537 61.8802 31.3128 61.8297 31.3633C61.7793 31.4137 61.7202 31.4381 61.6489 31.4381H59.1619C59.0906 31.4381 59.0315 31.4137 58.981 31.3633C58.9306 31.3146 58.9062 31.2537 58.9062 31.1824V16.8172C58.9062 16.7459 58.9306 16.6868 58.981 16.6363C59.0315 16.5859 59.0906 16.5615 59.1619 16.5615H61.6489C61.7758 16.5615 61.8749 16.6189 61.9462 16.732L65.1341 21.6833C65.1758 21.7685 65.2193 21.7685 65.261 21.6833L68.4697 16.732C68.541 16.6189 68.6402 16.5615 68.7671 16.5615Z" fill="currentColor"/>
<path d="M74.1764 31.3633C74.1259 31.3146 74.1016 31.2537 74.1016 31.1824V16.8172C74.1016 16.7459 74.1259 16.6868 74.1764 16.6363C74.2268 16.5859 74.2859 16.5615 74.3572 16.5615H76.8442C76.9155 16.5615 76.9746 16.5859 77.0251 16.6363C77.0737 16.6868 77.0998 16.7459 77.0998 16.8172V31.1824C77.0998 31.2537 77.0755 31.3128 77.0251 31.3633C76.9746 31.4137 76.9155 31.4381 76.8442 31.4381H74.3572C74.2859 31.4381 74.2268 31.4137 74.1764 31.3633Z" fill="currentColor"/>
<path d="M88.3066 16.6361C88.3553 16.5874 88.4162 16.5613 88.4875 16.5613H90.9744C91.0457 16.5613 91.1049 16.5857 91.1553 16.6361C91.204 16.6865 91.2301 16.7457 91.2301 16.817V31.1822C91.2301 31.2535 91.2057 31.3126 91.1553 31.363C91.1049 31.4135 91.0457 31.4378 90.9744 31.4378H88.5727C88.4301 31.4378 88.331 31.3822 88.2753 31.2674L82.771 22.1717C82.7431 22.13 82.7136 22.1109 82.6858 22.1178C82.6579 22.1248 82.644 22.1578 82.644 22.2135L82.6858 31.1805C82.6858 31.2518 82.6614 31.3109 82.611 31.3613C82.5606 31.4117 82.5014 31.4361 82.4301 31.4361H79.9431C79.8718 31.4361 79.8127 31.4117 79.7623 31.3613C79.7118 31.3126 79.6875 31.2518 79.6875 31.1805V16.8152C79.6875 16.7439 79.7118 16.6848 79.7623 16.6344C79.8127 16.5839 79.8718 16.5596 79.9431 16.5596H82.3449C82.4858 16.5596 82.5849 16.617 82.6423 16.73L88.124 25.7822C88.1518 25.8239 88.1797 25.8431 88.2092 25.8361C88.2371 25.8292 88.251 25.7978 88.251 25.7404L88.2301 16.8152C88.2301 16.7439 88.2545 16.6848 88.3049 16.6344L88.3066 16.6361Z" fill="currentColor"/>
<path d="M93.8951 31.3633C93.8446 31.3146 93.8203 31.2537 93.8203 31.1824V16.8172C93.8203 16.7459 93.8446 16.6868 93.8951 16.6363C93.9455 16.5859 94.0047 16.5615 94.076 16.5615H96.5629C96.6342 16.5615 96.6934 16.5859 96.7438 16.6363C96.7925 16.6868 96.8186 16.7459 96.8186 16.8172V31.1824C96.8186 31.2537 96.7942 31.3128 96.7438 31.3633C96.6934 31.4137 96.6342 31.4381 96.5629 31.4381H94.076C94.0047 31.4381 93.9455 31.4137 93.8951 31.3633Z" fill="currentColor"/>
<path d="M109.267 16.5615H111.754C111.825 16.5615 111.885 16.5859 111.935 16.6363C111.984 16.6868 112.01 16.7459 112.01 16.8172V31.1824C112.01 31.2537 111.985 31.3128 111.935 31.3633C111.885 31.4137 111.825 31.4381 111.754 31.4381H109.267C109.196 31.4381 109.137 31.4137 109.086 31.3633C109.036 31.3146 109.011 31.2537 109.011 31.1824V21.812C109.011 21.7563 108.998 21.7268 108.97 21.7268C108.942 21.7268 108.912 21.7476 108.885 21.7911L106.632 25.318C106.561 25.4311 106.462 25.4885 106.335 25.4885H105.081C104.954 25.4885 104.855 25.4328 104.784 25.318L102.531 21.7911C102.504 21.7494 102.474 21.7302 102.446 21.7372C102.418 21.7441 102.405 21.7772 102.405 21.8328V31.1824C102.405 31.2537 102.38 31.3128 102.33 31.3633C102.279 31.4137 102.22 31.4381 102.149 31.4381H99.6619C99.5906 31.4381 99.5315 31.4137 99.481 31.3633C99.4306 31.3146 99.4062 31.2537 99.4062 31.1824V16.8172C99.4062 16.7459 99.4306 16.6868 99.481 16.6363C99.5315 16.5859 99.5906 16.5615 99.6619 16.5615H102.149C102.276 16.5615 102.375 16.6189 102.446 16.732L105.634 21.6833C105.676 21.7685 105.719 21.7685 105.761 21.6833L108.97 16.732C109.041 16.6189 109.14 16.5615 109.267 16.5615Z" fill="currentColor"/>
<path d="M123.782 31.2241L123.144 29.1424C123.116 29.0867 123.079 29.0572 123.038 29.0572H117.81C117.768 29.0572 117.732 29.085 117.704 29.1424L117.088 31.2241C117.046 31.3668 116.954 31.4363 116.812 31.4363H114.112C114.027 31.4363 113.963 31.412 113.921 31.3615C113.879 31.3128 113.871 31.2381 113.9 31.1389L118.49 16.7737C118.532 16.6328 118.624 16.5615 118.766 16.5615H122.102C122.243 16.5615 122.335 16.6328 122.379 16.7737L126.968 31.1389C126.982 31.1668 126.989 31.2033 126.989 31.245C126.989 31.372 126.911 31.4363 126.756 31.4363H124.057C123.916 31.4363 123.824 31.365 123.78 31.2241H123.782ZM118.554 26.7407H122.295C122.38 26.7407 122.408 26.6989 122.38 26.6137L120.467 20.3024C120.453 20.2467 120.432 20.2207 120.403 20.2276C120.375 20.2346 120.352 20.2589 120.339 20.3024L118.469 26.6137C118.455 26.6989 118.483 26.7407 118.554 26.7407Z" fill="currentColor"/>
<path d="M128.222 31.353C128.18 31.2974 128.187 31.2261 128.243 31.1409L132.365 24.0643C132.393 24.0226 132.393 23.9791 132.365 23.9374L128.243 16.8609L128.201 16.7339C128.201 16.6209 128.28 16.5635 128.434 16.5635H131.133C131.274 16.5635 131.38 16.6209 131.452 16.7339L134.213 21.6C134.255 21.6852 134.299 21.6852 134.34 21.6L137.102 16.7339C137.173 16.6209 137.28 16.5635 137.42 16.5635H140.099C140.198 16.5635 140.269 16.5913 140.311 16.6487C140.353 16.7061 140.346 16.7756 140.29 16.8609L136.168 23.9374C136.154 23.9791 136.154 24.0226 136.168 24.0643L140.29 31.1409L140.332 31.2678C140.332 31.3809 140.253 31.4383 140.099 31.4383H137.42C137.278 31.4383 137.172 31.3826 137.102 31.2678L134.34 26.4226C134.299 26.3374 134.255 26.3374 134.213 26.4226L131.429 31.2678C131.358 31.3809 131.252 31.4383 131.111 31.4383H128.433C128.333 31.4383 128.262 31.4104 128.22 31.353H128.222Z" fill="currentColor"/>
<defs>
<linearGradient id="paint0_linear_17_483" x1="3.99826" y1="24" x2="51.6208" y2="24" gradientUnits="userSpaceOnUse">
<stop stop-color="#E21680"/>
<stop offset="1" stop-color="#FF633A"/>
</linearGradient>
</defs>
</svg>

</div>
<hr>

<div align="center" style="line-height: 1;">
  <a href="https://www.minimaxi.com/en" target="_blank" style="margin: 2px; color: var(--fgColor-default);">
    <img alt="Homepage" src="https://img.shields.io/badge/_Homepage-MiniMax-FF4040?style=flat-square&labelColor=2C3E50&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHhtbG5zOnhsaW5rPSJodHRwOi8vd3d3LnczLm9yZy8xOTk5L3hsaW5rIiB2aWV3Qm94PSIwIDAgNDkwLjE2IDQxMS43Ij48ZGVmcz48c3R5bGU+LmNscy0xe2ZpbGw6I2ZmZjt9PC9zdHlsZT48L2RlZnM+PHBhdGggY2xhc3M9ImNscy0xIiBkPSJNMjMzLjQ1LDQwLjgxYTE3LjU1LDE3LjU1LDAsMSwwLTM1LjEsMFYzMzEuNTZhNDAuODIsNDAuODIsMCwwLDEtODEuNjMsMFYxNDVhMTcuNTUsMTcuNTUsMCwxLDAtMzUuMDksMHY3OS4wNmE0MC44Miw0MC44MiwwLDAsMS04MS42MywwVjE5NS40MmExMS42MywxMS42MywwLDAsMSwyMy4yNiwwdjI4LjY2YTE3LjU1LDE3LjU1LDAsMCwwLDM1LjEsMFYxNDVBNDAuODIsNDAuODIsMCwwLDEsMTQwLDE0NVYzMzEuNTZhMTcuNTUsMTcuNTUsMCwwLDAsMzUuMSwwVjIxNy41aDBWNDAuODFhNDAuODEsNDAuODEsMCwxLDEsODEuNjIsMFYyODEuNTZhMTEuNjMsMTEuNjMsMCwxLDEtMjMuMjYsMFptMjE1LjksNjMuNEE0MC44Niw0MC44NiwwLDAsMCw0MDguNTMsMTQ1VjMwMC44NWExNy41NSwxNy41NSwwLDAsMS0zNS4wOSwwdi0yNjBhNDAuODIsNDAuODIsMCwwLDAtODEuNjMsMFYzNzAuODlhMTcuNTUsMTcuNTUsMCwwLDEtMzUuMSwwVjMzMGExMS42MywxMS42MywwLDEsMC0yMy4yNiwwdjQwLjg2YTQwLjgxLDQwLjgxLDAsMCwwLDgxLjYyLDBWNDAuODFhMTcuNTUsMTcuNTUsMCwwLDEsMzUuMSwwdjI2MGE0MC44Miw0MC44MiwwLDAsMCw4MS42MywwVjE0NWExNy41NSwxNy41NSwwLDEsMSwzNS4xLDBWMjgxLjU2YTExLjYzLDExLjYzLDAsMCwwLDIzLjI2LDBWMTQ1QTQwLjg1LDQwLjg1LDAsMCwwLDQ0OS4zNSwxMDQuMjFaIi8+PC9zdmc+&logoWidth=20" style="display: inline-block; vertical-align: middle;"/>
  </a>
  <a href="https://arxiv.org/abs/2501.08313" target="_blank" style="margin: 2px;">
    <img alt="Paper" src="https://img.shields.io/badge/📖_Paper-MiniMax--01-FF4040?style=flat-square&labelColor=2C3E50" style="display: inline-block; vertical-align: middle;"/>
  </a>
  <a href="https://huggingface.co/MiniMaxAI" target="_blank" style="margin: 2px;">
    <img alt="Hugging Face" src="https://img.shields.io/badge/🤗_Hugging_Face-MinMax-FF4040?style=flat-square&labelColor=2C3E50" style="display: inline-block; vertical-align: middle;"/>
  </a>
</div>
<div align="center" style="line-height: 1;">
  <a href="https://www.hailuo.ai/" target="_blank" style="margin: 2px;">
    <img alt="Chat" src="https://img.shields.io/badge/Chat-_Hailuo AI-FF4040?style=flat-square&labelColor=2C3E50&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHhtbG5zOnhsaW5rPSJodHRwOi8vd3d3LnczLm9yZy8xOTk5L3hsaW5rIiB2aWV3Qm94PSIwIDAgMzc1LjE0IDM3NS4xNCI+PGRlZnM+PHN0eWxlPi5jbHMtMXtmaWxsOnVybCgjdW5uYW1lZC1ncmFkaWVudCk7fTwvc3R5bGU+PGxpbmVhckdyYWRpZW50IGlkPSJ1bm5hbWVkLWdyYWRpZW50IiB4MT0iOC40MiIgeTE9IjEzLjgxIiB4Mj0iNDI5LjY1IiB5Mj0iNDIyLjM3IiBncmFkaWVudFVuaXRzPSJ1c2VyU3BhY2VPblVzZSI+PHN0b3Agb2Zmc2V0PSIwLjA5IiBzdG9wLWNvbG9yPSIjZmZhYjBjIi8+PHN0b3Agb2Zmc2V0PSIwLjMxIiBzdG9wLWNvbG9yPSIjZmY1NTM4Ii8+PHN0b3Agb2Zmc2V0PSIwLjQ2IiBzdG9wLWNvbG9yPSIjZTk0MDVkIi8+PHN0b3Agb2Zmc2V0PSIwLjc1IiBzdG9wLWNvbG9yPSIjZDI2NmRhIi8+PHN0b3Agb2Zmc2V0PSIwLjg5IiBzdG9wLWNvbG9yPSIjZDU4NGVmIi8+PC9saW5lYXJHcmFkaWVudD48L2RlZnM+PHBhdGggY2xhc3M9ImNscy0xIiBkPSJNMzc1LjE0LDE4Ny41N0MzNzUuMTQsODQsMjkwLjc0LS4yNiwxODcuMDksMCw4NC4yNi4yNi4yNiw4NC4yNSwwLDE4Ny4wOWMtLjI2LDEwMy42NSw4NCwxODgsMTg3LjU3LDE4OEgzMTAuODJBNjQuMjEsNjQuMjEsMCwwLDAsMzc1LDMxMC45M1YxOTMuODJoMEMzNzUuMDksMTkxLjc5LDM3NS4xNCwxODkuNjcsMzc1LjE0LDE4Ny41N1ptLTI4NCwxMDQuMTdjLTI5Ljg2LTI1LjQ5LTQ4LjI2LTY2LjI3LTQ3LjQtMTA3Ljg1cS4wOS00LjM4LjQ2LTguNzNWMTc1YzQuMzItNDkuNiwzNi4zNy05NS44OCw4MS4yOS0xMTcuMzZTMjI2LjUyLDQwLjIxLDI2Ny44NSw2OHM2Ni4zMiw3OC4yMSw2My40LDEyNy45MmExNzgsMTc4LDAsMCwxLTUuMTQsMzIuMjVjLTEsNC4yLTIuMyw4LjU3LTUuMjgsMTEuNzJzLTguMiw0LjYtMTEuNzMsMi4wOWMtMy4zNy0yLjQxLTMuODctNy4xMi00LjE2LTExLjI1LTIuMzMtMzMuMzctMTEuMjQtNjcuNzYtMzMuNzktOTIuNDdhMTAzLjY3LDEwMy42NywwLDAsMC02Ni4zOC0zMi44NEExMDcuMTksMTA3LjE5LDAsMCwwLDEzMy4yMiwxMjVDMTE2LDEzNy4yNywxMDIuNTUsMTU0Ljg4LDk2LDE3NXMtNS44Niw0Mi42MSwyLjcxLDYxLjkzYTgxLjg5LDgxLjg5LDAsMCwwLDI5LjcxLDM1YzIyLjk0LDE1LjA2LDU0LjMxLDE3LjIsNzguMTQsMy42czM4LjA3LTQzLjEsMzItNjkuODZTMjA1LjQsMTU4LDE3OC4xMSwxNjAuODRjLTQuMTYuNDMtMTAuMTMsMC0xMC4yOC00LjIxLS4xMi0zLjI0LDMuNzctNC45NCw3LTUuNTIsMjcuNjgtNSw1Ny4zNCw5LjA5LDcyLjUzLDMyLjc3czE2LDU1LjQxLDMuNTYsODAuNjYtMzcsNDMuNjktNjQuMzYsNTAuMzVDMTQ5LjY4LDMyMy44NywxMTYuMzEsMzEzLjI1LDkxLjExLDI5MS43NFoiLz48L3N2Zz4=&logoWidth=16" style="display: inline-block; vertical-align: middle;"/>
  </a>
  <a href="https://intl.minimaxi.com" style="margin: 2px;">
    <img alt="API" src="https://img.shields.io/badge/⚡_API-Platform-FF4040?style=flat-square&labelColor=2C3E50" style="display: inline-block; vertical-align: middle;"/>
  </a>
</div>
<div align="center" style="line-height: 1;">
  <a href="https://github.com/MiniMax-AI/MiniMax-01/blob/main/LICENSE" style="margin: 2px;">
    <img alt="License" src="https://img.shields.io/badge/📜_License-Model_Agreement-FF4040?style=flat-square&labelColor=2C3E50" style="display: inline-block; vertical-align: middle;"/>
  </a>
</div>


# MiniMax-Text-01

## 1. Introduction

MiniMax-Text-01 is a powerful language model with 456 billion total parameters, of which 45.9 billion are activated per token. To better unlock the long context capabilities of the model, MiniMax-Text-01 adopts a hybrid architecture that combines Lightning Attention, Softmax Attention and Mixture-of-Experts (MoE). Leveraging advanced parallel strategies and innovative compute-communication overlap methods—such as Linear Attention Sequence Parallelism Plus (LASP+), varlen ring attention, Expert Tensor Parallel (ETP), etc., MiniMax-Text-01's training context length is extended to 1 million tokens, and it can handle a context of up to 4 million tokens during the inference. On various academic benchmarks, MiniMax-Text-01 also demonstrates the performance of a top-tier model.

<p align="center">
  <img width="100%" src="figures/TextBench.png">
</p>

## 2. Model Architecture

The architecture of MiniMax-Text-01 is briefly described as follows:
- Total Parameters: 456B
- Activated Parameters per Token: 45.9B
- Number Layers: 80
- Hybrid Attention: a softmax attention is positioned after every 7 lightning attention.
  - Number of attention heads: 64
  - Attention head dimension: 128
- Mixture of Experts:
  - Number of experts: 32
  - Expert hidden dimension: 9216
  - Top-2 routing strategy
- Positional Encoding: Rotary Position Embedding (RoPE) applied to half of the attention head dimension with a base frequency of 10,000,000
- Hidden Size: 6144
- Vocab Size: 200,064

## 3. Evaluation

### Core Academic Benchmarks

| **Tasks**                     | **GPT-4o (11-20)** | **Claude-3.5-Sonnet (10-22)** | **Gemini-1.5-Pro (002)** | **Gemini-2.0-Flash (exp)** | **Qwen2.5-72B-Inst.** | **DeepSeek-V3** | **Llama-3.1-405B-Inst.** | **MiniMax-Text-01** |
|-------------------------------|--------------------|-------------------------------|--------------------------|----------------------------|-----------------------|-----------------|--------------------------|---------------------|
| **General**                   |                    |                               |                          |                            |                       |                 |                          |                     |
| MMLU<sup>*</sup>                      | 85.7               | 88.3                          | 86.8                     | 86.5                       | 86.1                  | 88.5        | **88.6**                 | 88.5                |
| MMLU-Pro<sup>*</sup>                  | 74.4               | **78.0**                      | 75.8                     | 76.4                       | 71.1                  | 75.9            | 73.3                     | 75.7                |
| SimpleQA                      | **39.0**           | 28.1                          | 23.4                     | 26.6                       | 10.3                  | 24.9            | 23.2                     | 23.7                |
| C-SimpleQA                    | 64.6               | 56.8                          | 59.4                     | 63.3                       | 52.2                  | 64.8            | 54.7                     | **67.4**            |
| IFEval _(avg)_                | 84.1               | **90.1**                      | 89.4                     | 88.4                       | 87.2                  | 87.3            | 86.4                     | 89.1                |
| Arena-Hard                    | **92.4**           | 87.6                          | 85.3                     | 72.7                       | 81.2                  | 91.4            | 63.5                     | 89.1                |
| **Reasoning**                 |                    |                               |                          |                            |                       |                 |                          |                     |
| GPQA<sup>*</sup> _(diamond)_          | 46.0               | **65.0**                      | 59.1                     | 62.1                       | 49.0                  | 59.1            | 50.7                     | 54.4                |
| DROP<sup>*</sup> _(F1)_               | 89.2               | 88.8                          | 89.2                     | 89.3                       | 85.0                  | 91.0        | **92.5**                 | 87.8                |
| **Mathematics**               |                    |                               |                          |                            |                       |                 |                          |                     |
| GSM8k<sup>*</sup>                     | 95.6               | **96.9**                      | 95.2                     | 95.4                       | 95.8                  | 96.7            | 96.7                     | 94.8                |
| MATH<sup>*</sup>                      | 76.6               | 74.1                          | **84.6**                 | 83.9                       | 81.8                  | **84.6**        | 73.8                     | 77.4                |
| **Coding**                    |                    |                               |                          |                            |                       |                 |                          |                     |
| MBPP +                        | 76.2               | 75.1                          | 75.4                     | 75.9                       | 77.0              | **78.8**        | 73.0                     | 71.7                |
| HumanEval                     | 90.2               | **93.7**                      | 86.6                     | 89.6                       | 86.6                  | 92.1            | 89.0                     | 86.9                |

<sup>*</sup> Evaluated following a _0-shot CoT_ setting.

### Long Benchmarks
#### 4M Needle In A Haystack Test
<p align="center">
  <img width="90%" src="figures/niah.png">
</p>

#### Ruler
| Model | 4k | 8k | 16k | 32k | 64k | 128k | 256k | 512k | 1M |
|-------|----|----|-----|-----|-----|------|------|------|----|
| **GPT-4o (11-20)** | **0.970** | 0.921 | 0.890 | 0.888 | 0.884 | - | - | - | - |
| **Claude-3.5-Sonnet (10-22)** | 0.965 | 0.960 | 0.957 | 0.950 | **0.952** | 0.938 | - | - | - |
| **Gemini-1.5-Pro (002)** | 0.962 | 0.960 | **0.960** | **0.958** | 0.938 | 0.917 | 0.916 | 0.861 | 0.850 |
| **Gemini-2.0-Flash (exp)** | 0.960 | 0.960 | 0.951 | 0.957 | 0.937 | 0.860 | 0.797 | 0.709 | - |
| **MiniMax-Text-01** | 0.963 | **0.961** | 0.953 | 0.954 | 0.943 | **0.947** | **0.945** | **0.928** | **0.910** |

#### LongBench v2
| **Model**                  | **overall** | **easy** | **hard** | **short** | **medium** | **long** |
|----------------------------|-------------|----------|----------|------------|------------|----------|
| Human                      | 53.7        | 100.0    | 25.1     | 47.2       | 59.1       | 53.7     |
| **w/ CoT**                 |             |          |          |            |            |          |
| GPT-4o (11-20)             | 51.4        | 54.2     | 49.7     | 59.6       | 48.6       | 43.5     |
| Claude-3.5-Sonnet (10-22)  | 46.7        | 55.2     | 41.5     | 53.9       | 41.9       | 44.4     |
| Deepseek-V3                | -           | -        | -        | -          | -          | -        |
| Qwen2.5-72B-Inst.          | 43.5        | 47.9     | 40.8     | 48.9       | 40.9       | 39.8     |
| **MiniMax-Text-01**        | **56.5**    | **66.1** | **50.5** | **61.7**   | **56.7**   | **47.2** |
| **w/o CoT**                |             |          |          |            |            |          |
| GPT-4o (11-20)             | 50.1        | 57.4     | 45.6     | 53.3       | 52.4       | 40.2     |
| Claude-3.5-Sonnet (10-22)  | 41.0        | 46.9     | 37.3     | 46.1       | 38.6       | 37.0     |
| Deepseek-V3                | 48.7        | -        | -        | -          | -          | -        |
| Qwen2.5-72B-Inst.          | 42.1        | 42.7     | 41.8     | 45.6       | 38.1       | **44.4** |
| **MiniMax-Text-01**        | **52.9**    | **60.9** | **47.9** | **58.9**   | **52.6**   | 43.5     |

#### MTOB
| **Context Type** | **no context** | **half book** | **full book** | **Δ half book** | **Δ full book** |
|------------------|----------------|---------------|---------------|------------------|-----------------|
| **eng → kalam (ChrF)** | | | | | |
| GPT-4o (11-20) | 9.90 | **54.30** | - | 44.40 | - |
| Claude-3.5-Sonnet (10-22) | 20.22 | 53.62 | 55.65 | 33.39 | 35.42 |
| Gemini-1.5-Pro (002) | 16.79 | 53.68 | **57.90** | 36.89 | 41.11 |
| Gemini-2.0-Flash (exp) | 12.20 | 49.50 | 53.30 | 37.30 | 41.10 |
| Qwen-Long | 16.55 | 48.48 | 45.94 | 31.92 | 29.39 |
| **MiniMax-Text-01** | 6.0 | 51.74 | 51.60 | **45.7** | **45.6** |
| **kalam → eng (BLEURT)** | | | | | |
| GPT-4o (11-20) | 33.20 | 58.30 | - | 25.10 | - |
| Claude-3.5-Sonnet (10-22) | 31.42 | 59.70 | 62.30 | 28.28 | 30.88 |
| Gemini-1.5-Pro (002) | 32.02 | **61.52** | **63.09** | **29.50** | **31.07** |
| Gemini-2.0-Flash (exp) | 33.80 | 57.50 | 57.00 | 23.70 | 23.20 |
| Qwen-Long | 30.13 | 53.14 | 32.15 | 23.01 | 2.02 |
| **MiniMax-Text-01** | 33.65 | 57.10 | 58.00 | 23.45 | 24.35 |


## 4. Quickstart
Here we provide a simple example of loading the tokenizer and model to generate content.
```python
from paddlenlp.transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, QuantoConfig, GenerationConfig

# load hf config
hf_config = AutoConfig.from_pretrained("MiniMaxAI/MiniMax-Text-01", trust_remote_code=True)

# quantization config, int8 is recommended
quantization_config =  QuantoConfig(
            weights="int8",
            modules_to_not_convert=[
                "lm_head",
                "embed_tokens",
            ] + [f"model.layers.{i}.coefficient" for i in range(hf_config.num_hidden_layers)]
            + [f"model.layers.{i}.block_sparse_moe.gate" for i in range(hf_config.num_hidden_layers)]
        )

# assume 8 GPUs
world_size = 8
layers_per_device = hf_config.num_hidden_layers // world_size
# set device map
device_map = {
    'model.embed_tokens': 'cuda:0',
    'model.norm': f'cuda:{world_size - 1}',
    'lm_head': f'cuda:{world_size - 1}'
}
for i in range(world_size):
    for j in range(layers_per_device):
        device_map[f'model.layers.{i * layers_per_device + j}'] = f'cuda:{i}'

# load tokenizer
tokenizer = AutoTokenizer.from_pretrained("MiniMaxAI/MiniMax-Text-01")
prompt = "Hello!"
messages = [
    {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant created by MiniMax based on MiniMax-Text-01 model."}]},
    {"role": "user", "content": [{"type": "text", "text": prompt}]},
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
# tokenize and move to device
model_inputs = tokenizer(text, return_tensors="pd")

# load bfloat16 model, move to device, and apply quantization
quantized_model = AutoModelForCausalLM.from_pretrained(
    "MiniMaxAI/MiniMax-Text-01",
    dtype="bfloat16",
    device_map=device_map,
    quantization_config=quantization_config,
    trust_remote_code=True,
    offload_buffers=True,
)

# generate response
generation_config = GenerationConfig(
    max_new_tokens=20,
    eos_token_id=200020,
    use_cache=True,
)
generated_ids = quantized_model.generate(**model_inputs, generation_config=generation_config)[0]
print(f"generated_ids: {generated_ids}")
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]
response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
```

## 5. Citation

```
@misc{minimax2025minimax01scalingfoundationmodels,
      title={MiniMax-01: Scaling Foundation Models with Lightning Attention}, 
      author={MiniMax and Aonian Li and Bangwei Gong and Bo Yang and Boji Shan and Chang Liu and Cheng Zhu and Chunhao Zhang and Congchao Guo and Da Chen and Dong Li and Enwei Jiao and Gengxin Li and Guojun Zhang and Haohai Sun and Houze Dong and Jiadai Zhu and Jiaqi Zhuang and Jiayuan Song and Jin Zhu and Jingtao Han and Jingyang Li and Junbin Xie and Junhao Xu and Junjie Yan and Kaishun Zhang and Kecheng Xiao and Kexi Kang and Le Han and Leyang Wang and Lianfei Yu and Liheng Feng and Lin Zheng and Linbo Chai and Long Xing and Meizhi Ju and Mingyuan Chi and Mozhi Zhang and Peikai Huang and Pengcheng Niu and Pengfei Li and Pengyu Zhao and Qi Yang and Qidi Xu and Qiexiang Wang and Qin Wang and Qiuhui Li and Ruitao Leng and Shengmin Shi and Shuqi Yu and Sichen Li and Songquan Zhu and Tao Huang and Tianrun Liang and Weigao Sun and Weixuan Sun and Weiyu Cheng and Wenkai Li and Xiangjun Song and Xiao Su and Xiaodong Han and Xinjie Zhang and Xinzhu Hou and Xu Min and Xun Zou and Xuyang Shen and Yan Gong and Yingjie Zhu and Yipeng Zhou and Yiran Zhong and Yongyi Hu and Yuanxiang Fan and Yue Yu and Yufeng Yang and Yuhao Li and Yunan Huang and Yunji Li and Yunpeng Huang and Yunzhi Xu and Yuxin Mao and Zehan Li and Zekang Li and Zewei Tao and Zewen Ying and Zhaoyang Cong and Zhen Qin and Zhenhua Fan and Zhihang Yu and Zhuo Jiang and Zijia Wu},
      year={2025},
      eprint={2501.08313},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2501.08313}, 
}
```

## 6. Chatbot & API
For general use and evaluation, we provide a [Chatbot](https://www.hailuo.ai/) with online search capabilities and the [online API](https://intl.minimaxi.com) for developers.

Contact us at [model@minimaxi.com](mailto:model@minimaxi.com).




## Model Files

- [LICENSE](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/LICENSE) (8.6 KB)

- [README.md](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/README.md) (26.5 KB)

- [config.json](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/config.json) (1.8 KB)

- [configuration.json](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/configuration.json) (73.0 B)

- [configuration_minimax_text_01.py](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/configuration_minimax_text_01.py) (7.2 KB)

- [figures/MiniMaxLogo.png](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/figures/MiniMaxLogo.png) (44.3 KB)

- [figures/TextBench.png](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/figures/TextBench.png) (410.6 KB)

- [figures/VisionBench.png](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/figures/VisionBench.png) (429.5 KB)

- [figures/hailuo.svg](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/figures/hailuo.svg) (1.5 KB)

- [figures/image.jpg](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/figures/image.jpg) (391.5 KB)

- [figures/minimax.svg](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/figures/minimax.svg) (1.2 KB)

- [figures/niah.png](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/figures/niah.png) (1.4 MB)

- [main.py](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/main.py) (3.5 KB)

- [merges.txt](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/merges.txt) (2.3 MB)

- [model-00000-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00000-of-00413.safetensors) (4.6 GB)

- [model-00001-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00001-of-00413.safetensors) (2.0 GB)

- [model-00002-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00002-of-00413.safetensors) (2.2 GB)

- [model-00003-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00003-of-00413.safetensors) (2.1 GB)

- [model-00004-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00004-of-00413.safetensors) (2.0 GB)

- [model-00005-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00005-of-00413.safetensors) (2.0 GB)

- [model-00006-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00006-of-00413.safetensors) (2.1 GB)

- [model-00008-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00008-of-00413.safetensors) (2.1 GB)

- [model-00009-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00009-of-00413.safetensors) (2.0 GB)

- [model-00010-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00010-of-00413.safetensors) (2.1 GB)

- [model-00011-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00011-of-00413.safetensors) (2.0 GB)

- [model-00012-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00012-of-00413.safetensors) (2.1 GB)

- [model-00013-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00013-of-00413.safetensors) (2.0 GB)

- [model-00014-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00014-of-00413.safetensors) (2.1 GB)

- [model-00015-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00015-of-00413.safetensors) (2.0 GB)

- [model-00016-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00016-of-00413.safetensors) (2.1 GB)

- [model-00017-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00017-of-00413.safetensors) (2.0 GB)

- [model-00018-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00018-of-00413.safetensors) (2.1 GB)

- [model-00019-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00019-of-00413.safetensors) (2.0 GB)

- [model-00020-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00020-of-00413.safetensors) (2.1 GB)

- [model-00021-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00021-of-00413.safetensors) (2.0 GB)

- [model-00022-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00022-of-00413.safetensors) (2.1 GB)

- [model-00023-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00023-of-00413.safetensors) (2.0 GB)

- [model-00024-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00024-of-00413.safetensors) (2.1 GB)

- [model-00025-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00025-of-00413.safetensors) (2.0 GB)

- [model-00026-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00026-of-00413.safetensors) (2.1 GB)

- [model-00027-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00027-of-00413.safetensors) (2.0 GB)

- [model-00028-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00028-of-00413.safetensors) (2.1 GB)

- [model-00029-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00029-of-00413.safetensors) (2.0 GB)

- [model-00030-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00030-of-00413.safetensors) (2.1 GB)

- [model-00031-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00031-of-00413.safetensors) (2.0 GB)

- [model-00032-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00032-of-00413.safetensors) (2.1 GB)

- [model-00033-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00033-of-00413.safetensors) (2.0 GB)

- [model-00034-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00034-of-00413.safetensors) (2.1 GB)

- [model-00035-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00035-of-00413.safetensors) (2.0 GB)

- [model-00036-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00036-of-00413.safetensors) (2.1 GB)

- [model-00037-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00037-of-00413.safetensors) (2.0 GB)

- [model-00038-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00038-of-00413.safetensors) (2.1 GB)

- [model-00039-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00039-of-00413.safetensors) (2.0 GB)

- [model-00040-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00040-of-00413.safetensors) (2.1 GB)

- [model-00041-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00041-of-00413.safetensors) (2.0 GB)

- [model-00042-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00042-of-00413.safetensors) (2.1 GB)

- [model-00043-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00043-of-00413.safetensors) (2.0 GB)

- [model-00044-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00044-of-00413.safetensors) (2.1 GB)

- [model-00045-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00045-of-00413.safetensors) (2.0 GB)

- [model-00046-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00046-of-00413.safetensors) (2.1 GB)

- [model-00047-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00047-of-00413.safetensors) (2.0 GB)

- [model-00048-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00048-of-00413.safetensors) (2.1 GB)

- [model-00049-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00049-of-00413.safetensors) (2.0 GB)

- [model-00050-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00050-of-00413.safetensors) (2.1 GB)

- [model-00051-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00051-of-00413.safetensors) (2.0 GB)

- [model-00052-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00052-of-00413.safetensors) (2.1 GB)

- [model-00053-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00053-of-00413.safetensors) (2.0 GB)

- [model-00054-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00054-of-00413.safetensors) (2.1 GB)

- [model-00055-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00055-of-00413.safetensors) (2.0 GB)

- [model-00056-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00056-of-00413.safetensors) (2.1 GB)

- [model-00057-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00057-of-00413.safetensors) (2.0 GB)

- [model-00058-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00058-of-00413.safetensors) (2.1 GB)

- [model-00059-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00059-of-00413.safetensors) (2.0 GB)

- [model-00060-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00060-of-00413.safetensors) (2.1 GB)

- [model-00061-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00061-of-00413.safetensors) (2.0 GB)

- [model-00062-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00062-of-00413.safetensors) (2.1 GB)

- [model-00063-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00063-of-00413.safetensors) (2.0 GB)

- [model-00064-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00064-of-00413.safetensors) (2.1 GB)

- [model-00065-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00065-of-00413.safetensors) (2.0 GB)

- [model-00066-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00066-of-00413.safetensors) (2.1 GB)

- [model-00067-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00067-of-00413.safetensors) (2.0 GB)

- [model-00068-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00068-of-00413.safetensors) (2.1 GB)

- [model-00069-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00069-of-00413.safetensors) (2.0 GB)

- [model-00070-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00070-of-00413.safetensors) (2.1 GB)

- [model-00071-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00071-of-00413.safetensors) (2.0 GB)

- [model-00072-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00072-of-00413.safetensors) (2.1 GB)

- [model-00073-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00073-of-00413.safetensors) (2.0 GB)

- [model-00074-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00074-of-00413.safetensors) (2.1 GB)

- [model-00075-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00075-of-00413.safetensors) (2.0 GB)

- [model-00076-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00076-of-00413.safetensors) (2.1 GB)

- [model-00077-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00077-of-00413.safetensors) (2.0 GB)

- [model-00078-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00078-of-00413.safetensors) (2.1 GB)

- [model-00079-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00079-of-00413.safetensors) (2.0 GB)

- [model-00080-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00080-of-00413.safetensors) (2.1 GB)

- [model-00081-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00081-of-00413.safetensors) (2.0 GB)

- [model-00082-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00082-of-00413.safetensors) (2.1 GB)

- [model-00083-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00083-of-00413.safetensors) (2.0 GB)

- [model-00084-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00084-of-00413.safetensors) (2.1 GB)

- [model-00085-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00085-of-00413.safetensors) (2.0 GB)

- [model-00086-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00086-of-00413.safetensors) (2.1 GB)

- [model-00087-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00087-of-00413.safetensors) (2.0 GB)

- [model-00088-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00088-of-00413.safetensors) (2.1 GB)

- [model-00089-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00089-of-00413.safetensors) (2.0 GB)

- [model-00090-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00090-of-00413.safetensors) (2.1 GB)

- [model-00091-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00091-of-00413.safetensors) (2.0 GB)

- [model-00092-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00092-of-00413.safetensors) (2.1 GB)

- [model-00093-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00093-of-00413.safetensors) (2.0 GB)

- [model-00094-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00094-of-00413.safetensors) (2.1 GB)

- [model-00095-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00095-of-00413.safetensors) (2.0 GB)

- [model-00096-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00096-of-00413.safetensors) (2.1 GB)

- [model-00097-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00097-of-00413.safetensors) (2.0 GB)

- [model-00098-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00098-of-00413.safetensors) (2.1 GB)

- [model-00099-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00099-of-00413.safetensors) (2.0 GB)

- [model-00100-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00100-of-00413.safetensors) (2.1 GB)

- [model-00101-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00101-of-00413.safetensors) (2.0 GB)

- [model-00102-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00102-of-00413.safetensors) (2.1 GB)

- [model-00103-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00103-of-00413.safetensors) (2.0 GB)

- [model-00104-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00104-of-00413.safetensors) (2.0 GB)

- [model-00105-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00105-of-00413.safetensors) (2.1 GB)

- [model-00106-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00106-of-00413.safetensors) (2.0 GB)

- [model-00107-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00107-of-00413.safetensors) (2.0 GB)

- [model-00108-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00108-of-00413.safetensors) (2.1 GB)

- [model-00109-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00109-of-00413.safetensors) (2.0 GB)

- [model-00110-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00110-of-00413.safetensors) (2.0 GB)

- [model-00111-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00111-of-00413.safetensors) (2.0 GB)

- [model-00112-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00112-of-00413.safetensors) (2.1 GB)

- [model-00113-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00113-of-00413.safetensors) (2.0 GB)

- [model-00114-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00114-of-00413.safetensors) (2.1 GB)

- [model-00115-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00115-of-00413.safetensors) (2.0 GB)

- [model-00116-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00116-of-00413.safetensors) (2.1 GB)

- [model-00117-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00117-of-00413.safetensors) (2.0 GB)

- [model-00118-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00118-of-00413.safetensors) (2.1 GB)

- [model-00119-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00119-of-00413.safetensors) (2.0 GB)

- [model-00120-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00120-of-00413.safetensors) (2.1 GB)

- [model-00121-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00121-of-00413.safetensors) (2.0 GB)

- [model-00122-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00122-of-00413.safetensors) (2.1 GB)

- [model-00123-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00123-of-00413.safetensors) (2.0 GB)

- [model-00124-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00124-of-00413.safetensors) (2.1 GB)

- [model-00125-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00125-of-00413.safetensors) (2.0 GB)

- [model-00126-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00126-of-00413.safetensors) (2.1 GB)

- [model-00127-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00127-of-00413.safetensors) (2.0 GB)

- [model-00128-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00128-of-00413.safetensors) (2.1 GB)

- [model-00129-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00129-of-00413.safetensors) (2.0 GB)

- [model-00130-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00130-of-00413.safetensors) (2.1 GB)

- [model-00131-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00131-of-00413.safetensors) (2.0 GB)

- [model-00132-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00132-of-00413.safetensors) (2.1 GB)

- [model-00133-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00133-of-00413.safetensors) (2.0 GB)

- [model-00134-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00134-of-00413.safetensors) (2.1 GB)

- [model-00135-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00135-of-00413.safetensors) (2.0 GB)

- [model-00136-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00136-of-00413.safetensors) (2.1 GB)

- [model-00137-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00137-of-00413.safetensors) (2.0 GB)

- [model-00138-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00138-of-00413.safetensors) (2.1 GB)

- [model-00139-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00139-of-00413.safetensors) (2.0 GB)

- [model-00140-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00140-of-00413.safetensors) (2.1 GB)

- [model-00141-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00141-of-00413.safetensors) (2.0 GB)

- [model-00142-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00142-of-00413.safetensors) (2.1 GB)

- [model-00143-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00143-of-00413.safetensors) (2.0 GB)

- [model-00144-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00144-of-00413.safetensors) (2.1 GB)

- [model-00145-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00145-of-00413.safetensors) (2.0 GB)

- [model-00146-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00146-of-00413.safetensors) (2.1 GB)

- [model-00147-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00147-of-00413.safetensors) (2.0 GB)

- [model-00148-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00148-of-00413.safetensors) (2.1 GB)

- [model-00149-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00149-of-00413.safetensors) (2.0 GB)

- [model-00150-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00150-of-00413.safetensors) (2.1 GB)

- [model-00151-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00151-of-00413.safetensors) (2.0 GB)

- [model-00152-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00152-of-00413.safetensors) (2.1 GB)

- [model-00153-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00153-of-00413.safetensors) (2.0 GB)

- [model-00154-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00154-of-00413.safetensors) (2.1 GB)

- [model-00155-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00155-of-00413.safetensors) (2.0 GB)

- [model-00156-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00156-of-00413.safetensors) (2.1 GB)

- [model-00157-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00157-of-00413.safetensors) (2.0 GB)

- [model-00158-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00158-of-00413.safetensors) (2.1 GB)

- [model-00159-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00159-of-00413.safetensors) (2.0 GB)

- [model-00160-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00160-of-00413.safetensors) (2.1 GB)

- [model-00161-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00161-of-00413.safetensors) (2.0 GB)

- [model-00162-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00162-of-00413.safetensors) (2.1 GB)

- [model-00163-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00163-of-00413.safetensors) (2.0 GB)

- [model-00164-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00164-of-00413.safetensors) (2.1 GB)

- [model-00165-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00165-of-00413.safetensors) (2.0 GB)

- [model-00166-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00166-of-00413.safetensors) (2.1 GB)

- [model-00167-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00167-of-00413.safetensors) (2.0 GB)

- [model-00168-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00168-of-00413.safetensors) (2.1 GB)

- [model-00169-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00169-of-00413.safetensors) (2.0 GB)

- [model-00170-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00170-of-00413.safetensors) (2.1 GB)

- [model-00171-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00171-of-00413.safetensors) (2.0 GB)

- [model-00172-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00172-of-00413.safetensors) (2.1 GB)

- [model-00173-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00173-of-00413.safetensors) (2.0 GB)

- [model-00174-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00174-of-00413.safetensors) (2.1 GB)

- [model-00175-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00175-of-00413.safetensors) (2.0 GB)

- [model-00176-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00176-of-00413.safetensors) (2.1 GB)

- [model-00177-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00177-of-00413.safetensors) (2.0 GB)

- [model-00178-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00178-of-00413.safetensors) (2.1 GB)

- [model-00179-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00179-of-00413.safetensors) (2.0 GB)

- [model-00180-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00180-of-00413.safetensors) (2.1 GB)

- [model-00181-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00181-of-00413.safetensors) (2.0 GB)

- [model-00182-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00182-of-00413.safetensors) (2.1 GB)

- [model-00183-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00183-of-00413.safetensors) (2.0 GB)

- [model-00184-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00184-of-00413.safetensors) (2.1 GB)

- [model-00185-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00185-of-00413.safetensors) (2.0 GB)

- [model-00186-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00186-of-00413.safetensors) (2.1 GB)

- [model-00187-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00187-of-00413.safetensors) (2.0 GB)

- [model-00188-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00188-of-00413.safetensors) (2.1 GB)

- [model-00189-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00189-of-00413.safetensors) (2.0 GB)

- [model-00190-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00190-of-00413.safetensors) (2.1 GB)

- [model-00191-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00191-of-00413.safetensors) (2.0 GB)

- [model-00192-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00192-of-00413.safetensors) (2.1 GB)

- [model-00193-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00193-of-00413.safetensors) (2.0 GB)

- [model-00194-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00194-of-00413.safetensors) (2.1 GB)

- [model-00195-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00195-of-00413.safetensors) (2.0 GB)

- [model-00196-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00196-of-00413.safetensors) (2.1 GB)

- [model-00197-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00197-of-00413.safetensors) (2.0 GB)

- [model-00198-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00198-of-00413.safetensors) (2.1 GB)

- [model-00199-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00199-of-00413.safetensors) (2.0 GB)

- [model-00200-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00200-of-00413.safetensors) (2.1 GB)

- [model-00201-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00201-of-00413.safetensors) (2.2 GB)

- [model-00202-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00202-of-00413.safetensors) (2.0 GB)

- [model-00203-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00203-of-00413.safetensors) (2.0 GB)

- [model-00204-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00204-of-00413.safetensors) (2.1 GB)

- [model-00205-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00205-of-00413.safetensors) (2.0 GB)

- [model-00206-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00206-of-00413.safetensors) (2.0 GB)

- [model-00207-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00207-of-00413.safetensors) (2.1 GB)

- [model-00208-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00208-of-00413.safetensors) (2.0 GB)

- [model-00209-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00209-of-00413.safetensors) (2.0 GB)

- [model-00210-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00210-of-00413.safetensors) (2.1 GB)

- [model-00211-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00211-of-00413.safetensors) (2.0 GB)

- [model-00212-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00212-of-00413.safetensors) (2.1 GB)

- [model-00213-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00213-of-00413.safetensors) (2.0 GB)

- [model-00214-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00214-of-00413.safetensors) (2.1 GB)

- [model-00215-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00215-of-00413.safetensors) (2.0 GB)

- [model-00216-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00216-of-00413.safetensors) (2.1 GB)

- [model-00217-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00217-of-00413.safetensors) (2.0 GB)

- [model-00218-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00218-of-00413.safetensors) (2.1 GB)

- [model-00219-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00219-of-00413.safetensors) (2.0 GB)

- [model-00220-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00220-of-00413.safetensors) (2.1 GB)

- [model-00221-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00221-of-00413.safetensors) (2.0 GB)

- [model-00222-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00222-of-00413.safetensors) (2.1 GB)

- [model-00223-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00223-of-00413.safetensors) (2.0 GB)

- [model-00224-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00224-of-00413.safetensors) (2.1 GB)

- [model-00225-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00225-of-00413.safetensors) (2.0 GB)

- [model-00226-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00226-of-00413.safetensors) (2.1 GB)

- [model-00227-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00227-of-00413.safetensors) (2.0 GB)

- [model-00228-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00228-of-00413.safetensors) (2.1 GB)

- [model-00229-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00229-of-00413.safetensors) (2.0 GB)

- [model-00230-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00230-of-00413.safetensors) (2.1 GB)

- [model-00231-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00231-of-00413.safetensors) (2.0 GB)

- [model-00232-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00232-of-00413.safetensors) (2.1 GB)

- [model-00233-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00233-of-00413.safetensors) (2.0 GB)

- [model-00234-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00234-of-00413.safetensors) (2.1 GB)

- [model-00235-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00235-of-00413.safetensors) (2.0 GB)

- [model-00236-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00236-of-00413.safetensors) (2.1 GB)

- [model-00237-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00237-of-00413.safetensors) (2.0 GB)

- [model-00238-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00238-of-00413.safetensors) (2.1 GB)

- [model-00239-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00239-of-00413.safetensors) (2.0 GB)

- [model-00240-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00240-of-00413.safetensors) (2.1 GB)

- [model-00241-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00241-of-00413.safetensors) (2.0 GB)

- [model-00242-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00242-of-00413.safetensors) (2.1 GB)

- [model-00243-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00243-of-00413.safetensors) (2.0 GB)

- [model-00244-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00244-of-00413.safetensors) (2.1 GB)

- [model-00245-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00245-of-00413.safetensors) (2.0 GB)

- [model-00246-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00246-of-00413.safetensors) (2.1 GB)

- [model-00247-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00247-of-00413.safetensors) (2.0 GB)

- [model-00248-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00248-of-00413.safetensors) (2.1 GB)

- [model-00249-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00249-of-00413.safetensors) (2.0 GB)

- [model-00250-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00250-of-00413.safetensors) (2.1 GB)

- [model-00251-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00251-of-00413.safetensors) (2.0 GB)

- [model-00252-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00252-of-00413.safetensors) (2.1 GB)

- [model-00253-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00253-of-00413.safetensors) (2.0 GB)

- [model-00254-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00254-of-00413.safetensors) (2.1 GB)

- [model-00255-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00255-of-00413.safetensors) (2.0 GB)

- [model-00256-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00256-of-00413.safetensors) (2.1 GB)

- [model-00257-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00257-of-00413.safetensors) (2.0 GB)

- [model-00258-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00258-of-00413.safetensors) (2.1 GB)

- [model-00259-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00259-of-00413.safetensors) (2.0 GB)

- [model-00260-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00260-of-00413.safetensors) (2.1 GB)

- [model-00261-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00261-of-00413.safetensors) (2.0 GB)

- [model-00262-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00262-of-00413.safetensors) (2.1 GB)

- [model-00263-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00263-of-00413.safetensors) (2.0 GB)

- [model-00264-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00264-of-00413.safetensors) (2.1 GB)

- [model-00265-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00265-of-00413.safetensors) (2.0 GB)

- [model-00266-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00266-of-00413.safetensors) (2.1 GB)

- [model-00267-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00267-of-00413.safetensors) (2.0 GB)

- [model-00268-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00268-of-00413.safetensors) (2.1 GB)

- [model-00269-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00269-of-00413.safetensors) (2.0 GB)

- [model-00270-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00270-of-00413.safetensors) (2.1 GB)

- [model-00271-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00271-of-00413.safetensors) (2.0 GB)

- [model-00272-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00272-of-00413.safetensors) (2.1 GB)

- [model-00273-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00273-of-00413.safetensors) (2.0 GB)

- [model-00274-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00274-of-00413.safetensors) (2.1 GB)

- [model-00275-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00275-of-00413.safetensors) (2.0 GB)

- [model-00276-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00276-of-00413.safetensors) (2.1 GB)

- [model-00277-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00277-of-00413.safetensors) (2.0 GB)

- [model-00278-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00278-of-00413.safetensors) (2.1 GB)

- [model-00279-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00279-of-00413.safetensors) (2.0 GB)

- [model-00280-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00280-of-00413.safetensors) (2.1 GB)

- [model-00281-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00281-of-00413.safetensors) (2.0 GB)

- [model-00282-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00282-of-00413.safetensors) (2.1 GB)

- [model-00283-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00283-of-00413.safetensors) (2.0 GB)

- [model-00284-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00284-of-00413.safetensors) (2.1 GB)

- [model-00285-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00285-of-00413.safetensors) (2.0 GB)

- [model-00286-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00286-of-00413.safetensors) (2.1 GB)

- [model-00287-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00287-of-00413.safetensors) (2.0 GB)

- [model-00288-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00288-of-00413.safetensors) (2.1 GB)

- [model-00289-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00289-of-00413.safetensors) (2.0 GB)

- [model-00290-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00290-of-00413.safetensors) (2.1 GB)

- [model-00291-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00291-of-00413.safetensors) (2.0 GB)

- [model-00292-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00292-of-00413.safetensors) (2.1 GB)

- [model-00293-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00293-of-00413.safetensors) (2.0 GB)

- [model-00294-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00294-of-00413.safetensors) (2.1 GB)

- [model-00295-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00295-of-00413.safetensors) (2.0 GB)

- [model-00296-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00296-of-00413.safetensors) (2.1 GB)

- [model-00297-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00297-of-00413.safetensors) (2.0 GB)

- [model-00298-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00298-of-00413.safetensors) (2.1 GB)

- [model-00299-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00299-of-00413.safetensors) (2.0 GB)

- [model-00300-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00300-of-00413.safetensors) (2.1 GB)

- [model-00301-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00301-of-00413.safetensors) (2.0 GB)

- [model-00302-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00302-of-00413.safetensors) (2.1 GB)

- [model-00303-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00303-of-00413.safetensors) (2.0 GB)

- [model-00304-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00304-of-00413.safetensors) (2.1 GB)

- [model-00305-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00305-of-00413.safetensors) (2.0 GB)

- [model-00306-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00306-of-00413.safetensors) (2.1 GB)

- [model-00307-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00307-of-00413.safetensors) (2.0 GB)

- [model-00308-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00308-of-00413.safetensors) (2.1 GB)

- [model-00309-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00309-of-00413.safetensors) (2.0 GB)

- [model-00310-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00310-of-00413.safetensors) (2.1 GB)

- [model-00311-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00311-of-00413.safetensors) (2.0 GB)

- [model-00312-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00312-of-00413.safetensors) (2.1 GB)

- [model-00313-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00313-of-00413.safetensors) (2.0 GB)

- [model-00314-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00314-of-00413.safetensors) (2.0 GB)

- [model-00315-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00315-of-00413.safetensors) (2.0 GB)

- [model-00316-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00316-of-00413.safetensors) (2.0 GB)

- [model-00317-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00317-of-00413.safetensors) (2.0 GB)

- [model-00318-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00318-of-00413.safetensors) (2.1 GB)

- [model-00319-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00319-of-00413.safetensors) (2.0 GB)

- [model-00320-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00320-of-00413.safetensors) (2.0 GB)

- [model-00321-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00321-of-00413.safetensors) (4.1 GB)

- [model-00322-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00322-of-00413.safetensors) (2.1 GB)

- [model-00323-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00323-of-00413.safetensors) (2.0 GB)

- [model-00324-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00324-of-00413.safetensors) (2.1 GB)

- [model-00325-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00325-of-00413.safetensors) (2.0 GB)

- [model-00326-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00326-of-00413.safetensors) (2.1 GB)

- [model-00327-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00327-of-00413.safetensors) (2.0 GB)

- [model-00328-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00328-of-00413.safetensors) (2.1 GB)

- [model-00329-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00329-of-00413.safetensors) (2.0 GB)

- [model-00330-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00330-of-00413.safetensors) (2.1 GB)

- [model-00331-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00331-of-00413.safetensors) (2.0 GB)

- [model-00332-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00332-of-00413.safetensors) (2.1 GB)

- [model-00333-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00333-of-00413.safetensors) (2.0 GB)

- [model-00334-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00334-of-00413.safetensors) (2.1 GB)

- [model-00335-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00335-of-00413.safetensors) (2.0 GB)

- [model-00336-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00336-of-00413.safetensors) (2.1 GB)

- [model-00337-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00337-of-00413.safetensors) (2.0 GB)

- [model-00338-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00338-of-00413.safetensors) (2.1 GB)

- [model-00339-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00339-of-00413.safetensors) (2.0 GB)

- [model-00340-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00340-of-00413.safetensors) (2.1 GB)

- [model-00341-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00341-of-00413.safetensors) (2.0 GB)

- [model-00342-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00342-of-00413.safetensors) (2.1 GB)

- [model-00343-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00343-of-00413.safetensors) (2.0 GB)

- [model-00344-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00344-of-00413.safetensors) (2.1 GB)

- [model-00345-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00345-of-00413.safetensors) (2.0 GB)

- [model-00346-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00346-of-00413.safetensors) (2.1 GB)

- [model-00347-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00347-of-00413.safetensors) (2.0 GB)

- [model-00348-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00348-of-00413.safetensors) (2.1 GB)

- [model-00349-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00349-of-00413.safetensors) (2.0 GB)

- [model-00350-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00350-of-00413.safetensors) (2.1 GB)

- [model-00351-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00351-of-00413.safetensors) (2.0 GB)

- [model-00352-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00352-of-00413.safetensors) (2.1 GB)

- [model-00353-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00353-of-00413.safetensors) (2.0 GB)

- [model-00354-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00354-of-00413.safetensors) (2.1 GB)

- [model-00355-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00355-of-00413.safetensors) (2.0 GB)

- [model-00356-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00356-of-00413.safetensors) (2.1 GB)

- [model-00357-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00357-of-00413.safetensors) (2.0 GB)

- [model-00358-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00358-of-00413.safetensors) (2.1 GB)

- [model-00359-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00359-of-00413.safetensors) (2.0 GB)

- [model-00360-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00360-of-00413.safetensors) (2.1 GB)

- [model-00361-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00361-of-00413.safetensors) (2.0 GB)

- [model-00362-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00362-of-00413.safetensors) (2.1 GB)

- [model-00363-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00363-of-00413.safetensors) (2.0 GB)

- [model-00364-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00364-of-00413.safetensors) (2.1 GB)

- [model-00365-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00365-of-00413.safetensors) (2.0 GB)

- [model-00366-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00366-of-00413.safetensors) (2.1 GB)

- [model-00367-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00367-of-00413.safetensors) (2.0 GB)

- [model-00368-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00368-of-00413.safetensors) (2.1 GB)

- [model-00369-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00369-of-00413.safetensors) (2.0 GB)

- [model-00370-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00370-of-00413.safetensors) (2.1 GB)

- [model-00371-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00371-of-00413.safetensors) (2.0 GB)

- [model-00372-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00372-of-00413.safetensors) (2.1 GB)

- [model-00373-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00373-of-00413.safetensors) (2.0 GB)

- [model-00374-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00374-of-00413.safetensors) (2.1 GB)

- [model-00375-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00375-of-00413.safetensors) (2.0 GB)

- [model-00376-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00376-of-00413.safetensors) (2.1 GB)

- [model-00377-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00377-of-00413.safetensors) (2.0 GB)

- [model-00378-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00378-of-00413.safetensors) (2.1 GB)

- [model-00379-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00379-of-00413.safetensors) (2.0 GB)

- [model-00380-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00380-of-00413.safetensors) (2.1 GB)

- [model-00381-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00381-of-00413.safetensors) (2.0 GB)

- [model-00382-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00382-of-00413.safetensors) (2.1 GB)

- [model-00383-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00383-of-00413.safetensors) (2.0 GB)

- [model-00384-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00384-of-00413.safetensors) (2.1 GB)

- [model-00385-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00385-of-00413.safetensors) (2.0 GB)

- [model-00386-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00386-of-00413.safetensors) (2.1 GB)

- [model-00387-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00387-of-00413.safetensors) (2.0 GB)

- [model-00388-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00388-of-00413.safetensors) (2.1 GB)

- [model-00389-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00389-of-00413.safetensors) (2.0 GB)

- [model-00390-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00390-of-00413.safetensors) (2.1 GB)

- [model-00391-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00391-of-00413.safetensors) (2.0 GB)

- [model-00392-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00392-of-00413.safetensors) (2.1 GB)

- [model-00393-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00393-of-00413.safetensors) (2.0 GB)

- [model-00394-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00394-of-00413.safetensors) (2.1 GB)

- [model-00395-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00395-of-00413.safetensors) (2.0 GB)

- [model-00396-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00396-of-00413.safetensors) (2.1 GB)

- [model-00397-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00397-of-00413.safetensors) (2.0 GB)

- [model-00398-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00398-of-00413.safetensors) (2.1 GB)

- [model-00399-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00399-of-00413.safetensors) (2.0 GB)

- [model-00400-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00400-of-00413.safetensors) (2.1 GB)

- [model-00401-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00401-of-00413.safetensors) (2.0 GB)

- [model-00402-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00402-of-00413.safetensors) (2.1 GB)

- [model-00403-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00403-of-00413.safetensors) (2.0 GB)

- [model-00404-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00404-of-00413.safetensors) (2.1 GB)

- [model-00405-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00405-of-00413.safetensors) (2.0 GB)

- [model-00406-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00406-of-00413.safetensors) (2.1 GB)

- [model-00407-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00407-of-00413.safetensors) (2.0 GB)

- [model-00408-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00408-of-00413.safetensors) (2.1 GB)

- [model-00409-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00409-of-00413.safetensors) (2.0 GB)

- [model-00410-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00410-of-00413.safetensors) (2.1 GB)

- [model-00411-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00411-of-00413.safetensors) (2.0 GB)

- [model-00412-of-00413.safetensors](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model-00412-of-00413.safetensors) (1.3 GB)

- [model.safetensors.index.json](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/model.safetensors.index.json) (771.2 KB)

- [modeling_minimax_text_01.py](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/modeling_minimax_text_01.py) (73.5 KB)

- [tokenizer.json](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/tokenizer.json) (9.3 MB)

- [tokenizer_config.json](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/tokenizer_config.json) (984.0 B)

- [vocab.json](https://paddlenlp.bj.bcebos.com/models/community/MiniMaxAI/MiniMax-Text-01/vocab.json) (4.5 MB)


[Back to Main](../../)