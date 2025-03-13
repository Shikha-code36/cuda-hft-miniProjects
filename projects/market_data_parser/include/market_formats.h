#ifndef MARKET_FORMATS_H
#define MARKET_FORMATS_H

#include <cstdint>
#include <string>

namespace market {

// Common structures for financial market data protocols

// Message types for NASDAQ ITCH 5.0 protocol
enum class ItchMessageType : char {
    SYSTEM_EVENT = 'S',
    STOCK_DIRECTORY = 'R',
    TRADING_ACTION = 'H',
    REG_SHO_RESTRICTION = 'Y',
    MARKET_PARTICIPANT_POSITION = 'L',
    MWCB_DECLINE_LEVEL = 'V',
    MWCB_STATUS = 'W',
    IPO_QUOTING_PERIOD_UPDATE = 'K',
    ADD_ORDER = 'A',
    ADD_ORDER_WITH_MPID = 'F',
    ORDER_EXECUTED = 'E',
    ORDER_EXECUTED_WITH_PRICE = 'C',
    ORDER_CANCEL = 'X',
    ORDER_DELETE = 'D',
    ORDER_REPLACE = 'U',
    TRADE = 'P',
    CROSS_TRADE = 'Q',
    BROKEN_TRADE = 'B',
    NOII = 'I',
    RPII = 'N'
};

// Message types for NYSE PITCH protocol
enum class PitchMessageType : char {
    ADD_ORDER = 'A',
    ORDER_EXECUTED = 'E',
    ORDER_CANCEL = 'X',
    TRADE = 'P',
    ORDER_DELETE = 'D',
    ORDER_REPLACE = 'U',
    TRADE_BREAK = 'B',
    TRADING_STATUS = 'H',
    AUCTION_UPDATE = 'I',
    AUCTION_SUMMARY = 'J',
    RETAIL_PRICE_IMPROVEMENT = 'R',
    STOCK_SUMMARY = 'G'
};

// Common header format for market data messages
struct MessageHeader {
    uint16_t length;       // Message length (including header)
    uint64_t timestamp;    // Nanoseconds since midnight
    char type;             // Message type (see enums above)
};

// ITCH 5.0 specific message structures

// System Event Message
struct SystemEventMessage {
    MessageHeader header;
    uint16_t stock_locate;
    uint16_t tracking_number;
    char event_code;
};

// Add Order Message
struct AddOrderMessage {
    MessageHeader header;
    uint16_t stock_locate;
    uint16_t tracking_number;
    uint64_t order_reference_number;
    char buy_sell_indicator;
    uint32_t shares;
    char stock[8];
    uint32_t price;
    char mpid[4];  // Only used in AddOrderWithMPID messages
};

// Order Executed Message
struct OrderExecutedMessage {
    MessageHeader header;
    uint16_t stock_locate;
    uint16_t tracking_number;
    uint64_t order_reference_number;
    uint32_t executed_shares;
    uint64_t match_number;
};

// Order Delete Message
struct OrderDeleteMessage {
    MessageHeader header;
    uint16_t stock_locate;
    uint16_t tracking_number;
    uint64_t order_reference_number;
};

// NYSE PITCH specific message structures

// PITCH Add Order Message
struct PitchAddOrderMessage {
    MessageHeader header;
    uint64_t order_id;
    char side;
    uint32_t shares;
    char symbol[8];
    uint32_t price;
    char display;
    char participant_id[4];
};

// PITCH Order Executed Message
struct PitchOrderExecutedMessage {
    MessageHeader header;
    uint64_t order_id;
    uint32_t executed_shares;
    uint64_t execution_id;
    uint32_t price;
};

// Message parsing results
struct ParseResult {
    bool success;
    std::string error_message;
    size_t bytes_processed;
};

}  // namespace market

#endif // MARKET_FORMATS_H